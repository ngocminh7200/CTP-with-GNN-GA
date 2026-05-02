import networkx as nx
import numpy as np
from Instance import Instance
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class Graph:

    def __init__(self):
        self.G = nx.Graph()
        self.ds = None
        self.sched = None
        self.data = None

    def _check_constraints(self):
        """Compute all constraint violations for the current schedule.
        Results are cached as instance attributes for use by feature builders.
        """
        ds, sc = self.ds, self.sched
        C, R, T, P, G = self.ds.get_sizes()

        # --- H2: Professor conflict ---
        all_profs = ds.professors['P-Index'].drop_duplicates().tolist()
        self.prof_ts = (
            pd.crosstab(sc['P-Index'], sc['timeslot'])
            .reindex(index=all_profs, columns=ds.timeslots, fill_value=0)
        )

        # Per-course: does this course cause a professor conflict?
        self.h2_course = np.zeros(C, dtype=np.float32)
        for c in range(C):
            prf = sc.iloc[c]['P-Index']
            ts = sc.iloc[c]['timeslot']
            if self.prof_ts.loc[prf, ts] > 1:
                self.h2_course[c] = 1

        # --- H3: Room conflict ---
        all_rooms = ds.rooms['Host Key'].drop_duplicates().tolist()
        self.room_ts = (
            pd.crosstab(sc['Room'], sc['timeslot'])
            .reindex(index=all_rooms, columns=ds.timeslots, fill_value=0)
        )

        self.h3_course = np.zeros(C, dtype=np.float32)
        for c in range(C):
            rm = sc.iloc[c]['Room']
            ts = sc.iloc[c]['timeslot']
            if self.room_ts.loc[rm, ts] > 1:
                self.h3_course[c] = 1.0

        # --- H4: Room capacity ---
        # Binary: is s_c > c_r for the assigned room?
        self.h4_course = np.zeros(C, dtype=np.float32)
        self.h4_overflow = np.zeros(C, dtype=np.float32)
        for c in range(C):
            overflow = sc.iloc[c]['NUMSTU'] - sc.iloc[c]['Capacity']
            if overflow > 0:
                self.h4_course[c] = 1.0
                self.h4_overflow[c] = overflow

        max_overflow = max(self.h4_overflow.max(), 1.0)
        self.h4_overflow_norm = self.h4_overflow / max_overflow

        # --- S1: Global timeslot crowding ---
        # courses_at_t[t] = total courses at timeslot t
        self.courses_at_t = sc.groupby('timeslot').size()

        # --- S2: Program overlap ---
        all_programs = sc['dl'].unique().tolist()
        self.prog_ts = (
            pd.crosstab(sc['dl'], sc['timeslot'])
            .reindex(index=all_programs, columns=ds.timeslots, fill_value=0)
        )

        # s^overlap_{g,t} = max(0, prog_ts - 1)
        self.s2_overlap = np.maximum(0, self.prog_ts - 1).astype(np.float32)

        # --- Precompute room occupancy matrix as numpy for fast lookup ---
        # room_occupied[r_idx, t_idx] = number of courses in room r at timeslot t
        self._room_occupied = self.Room_ts.to_numpy().astype(np.float32)
        self._room_list = all_rooms
        self._room_capacities = np.array(
            [ds.rooms.loc[ds.rooms['Host Key'] == r, 'Capacity'].iloc[0]
             if r in ds.rooms['Host Key'].values else 0
             for r in all_rooms], dtype=np.float32
        )

    def _build_course_features(self) -> np.ndarray:
        """Course node features (C, 12).


        The per-timeslot information (which slots are free, which rooms
        are available) is already carried by the timeslot and room nodes
        and flows to course nodes through 4 layers of message passing.
        Encoding it directly in course features short-circuits the GNN
        and hardcodes T=25.

        Constraint violation features [0-3]:
            [0] h2_violation        — professor conflict flag (binary)
            [1] h3_violation        — room conflict flag (binary)
            [2] h4_violation        — capacity violation flag (binary)
            [3] h4_overflow_norm    — normalized capacity overflow degree

        Room/capacity context [4-5]:
            [4] room_fit            — assigned room capacity / enrollment (clipped ≤ 3)
            [5] flexibility         — fraction of ALL rooms that can fit this course

        Scheduling context [6-9]:
            [6] feasibility_ratio   — fraction of timeslots where both a suitable
                                      room is free AND the professor is free
            [7] prof_load_norm      — professor's total active slots / T
            [8] current_ts_norm     — current timeslot index / T
            [9] enrollment_norm     — enrollment / max enrollment across all courses

        Soft constraint signals [10-11]:
            [10] s2_exposure        — program overlap count at current timeslot,
                                      normalized (how many same-program courses
                                      share this course's timeslot)
            [11] crowding_norm      — total courses at current timeslot / C
                                      (how crowded the current timeslot is)
        """
        # TODO: remove [4], [6], adjust [8] (cuz for us, a course only belong to 1 prog)
        ds, sc = self.ds, self.sched
        C = len(ds.course_list)
        T = len(ds.timeslots)
        timeslot_list = list(ds.timeslots)

        # --- [4] Room fit ---
        room_fit = (sc['Capacity'] / sc['NUMSTU']).to_numpy()
        room_fit = np.clip(room_fit, 0.0, 3.0)

        # --- [5] Flexibility ---
        flexibility = np.array(
            [(ds.rooms['Capacity'] >= ds.courses.iloc[c]['NUMSTU']).mean()
             for c in range(C)], dtype=np.float32
        )

        # --- [6] Feasibility ratio (summary scalar, not per-timeslot) ---
        feasibility = np.zeros(C, dtype=np.float32)
        for c in range(C):
            enrollment = sc.iloc[c]['NUMSTU']
            prf = sc.iloc[c]['P-Index']
            current_ts = sc.iloc[c]['timeslot']
            current_room = sc.iloc[c]['Room']

            suitable_mask = self._room_capacities >= enrollment
            num_suitable = max(suitable_mask.sum(), 1)

            current_room_idx = None
            if current_room in self._room_list:
                current_room_idx = self._room_list.index(current_room)

            free_count = 0
            for t_idx, t in enumerate(timeslot_list):
                # Check professor free
                prof_load = self.prof_ts.loc[prf, t]
                if t == current_ts:
                    prof_load -= 1
                if prof_load > 0:
                    continue

                # Check room availability
                occupancy = self._room_occupied[:, t_idx].copy()
                if t == current_ts and current_room_idx is not None:
                    occupancy[current_room_idx] -= 1
                if ((occupancy == 0) & suitable_mask).any():
                    free_count += 1

            feasibility[c] = free_count / T

        # --- [7] Professor load ---
        prof_load_norm = np.zeros(C, dtype=np.float32)
        for c in range(C):
            prf = sc.iloc[c]['P-Index']
            active = (self.prof_ts.loc[prf] > 0).sum()
            prof_load_norm[c] = active / T

        # --- [8] Current timeslot normalized ---
        timeslot_to_idx = {t: i for i, t in enumerate(timeslot_list)}
        current_ts_norm = np.array(
            [timeslot_to_idx.get(sc.iloc[c]['timeslot'], 0) / T
             for c in range(C)], dtype=np.float32
        )

        # --- [9] Enrollment normalized ---
        enrollments = sc['NUMSTU'].to_numpy().astype(np.float32)
        max_enroll = max(enrollments.max(), 1.0)
        enrollment_norm = enrollments / max_enroll

        # --- [10] S2 program overlap exposure ---
        s2_exposure = np.zeros(C, dtype=np.float32)
        for c in range(C):
            prog = sc.iloc[c]['dl']
            ts = sc.iloc[c]['timeslot']
            if prog in self.prog_ts.index and ts in self.prog_ts.columns:
                overlap = max(0, self.prog_ts.loc[prog, ts] - 1)
                s2_exposure[c] = overlap
        max_s2 = max(s2_exposure.max(), 1.0)
        s2_exposure = s2_exposure / max_s2

        # --- [11] Timeslot crowding ---
        crowding = np.zeros(C, dtype=np.float32)
        for c in range(C):
            ts = sc.iloc[c]['timeslot']
            crowding[c] = self.courses_at_t.get(ts, 0)
        crowding_norm = crowding / max(C, 1)

        return np.stack([
            self.h2_course,  # [0]  professor conflict
            self.h3_course,  # [1]  room conflict
            self.h4_course,  # [2]  capacity violation
            self.h4_overflow_norm,  # [3]  overflow degree
            room_fit,  # [4]  room fit ratio
            flexibility,  # [5]  room flexibility
            feasibility,  # [6]  feasibility ratio (scalar summary)
            prof_load_norm,  # [7]  professor busyness
            current_ts_norm,  # [8]  current timeslot
            enrollment_norm,  # [9]  enrollment size
            s2_exposure,  # [10] program overlap
            crowding_norm,  # [11] timeslot crowding
        ], axis=1).astype(np.float32)

    # above has been fixed

    def _build_professor_features(self) -> np.ndarray:
        """Professor node features (P, 3).

        Professors participate in H2 (conflict).

        [0] h2_any_conflict      — binary: any timeslot conflict?
        [1] h2_conflict_ratio    — fraction of timeslots with conflicts
        [2] num_courses_norm     — |C_p| / max|C_p|
        [3] teaching_density     — actual slots used / T (how spread out)
        """
        # TODO: question about [1], remove [2]
        # remove[1], what is this?
        ds = self.ds
        T = len(ds.timeslots)

        conflicts = (self.prof_ts > 1).astype(np.float32)
        h2_any = (conflicts.sum(axis=1).to_numpy() > 0).astype(np.float32)

        course_counts = np.array(
            [len(ds.C_p.get(p, [])) if isinstance(ds.C_p, dict) else len(ds.C_p[p]) if p in ds.C_p else 0
             for p in self.prof_ts.index.tolist()],
            dtype=np.float32
        )

        active_slots = (self.prof_ts > 0).astype(np.float32).sum(axis=1).to_numpy()
        teaching_density = active_slots / T

        return np.stack([
            h2_any,  # [0]
            course_counts,  # [2]??
            teaching_density,  # [3]
        ], axis=1).astype(np.float32)

    # above fixed

    def _build_room_features(self) -> np.ndarray:
        """Room node features (R, 5).

        [0] h3_any_conflict
        [1] h3_conflict_ratio
        [2] occupancy
        [3] capacity_norm — room capacity / max capacity
        [4] demand_ratio  — how many courses need this room / total timeslots
        """
        # TODO: remove [3]
        # remove [4] as well. goodbye~
        ds = self.ds
        T = len(ds.timeslots)

        conflicts = (self.room_ts > 1).astype(np.float32)
        h3_any = (conflicts.sum(axis=1).to_numpy() > 0).astype(np.float32)
        h3_ratio = conflicts.sum(axis=1).to_numpy() / T

        occupancy = (self.room_ts > 0).astype(np.float32).sum(axis=1).to_numpy() / T

        max_cap = max(self._room_capacities.max(), 1.0)
        capacity_norm = self._room_capacities / max_cap

        total_assigned = self.room_ts.sum(axis=1).to_numpy().astype(np.float32)
        demand_ratio = total_assigned / T

        return np.stack([
            h3_any,  # [0]
            h3_ratio,  # [1]
            occupancy,  # [2]
            capacity_norm,  # [3]
            demand_ratio,  # [4]
        ], axis=1).astype(np.float32)

    def _build_timeslot_features(self) -> np.ndarray:
        """Timeslot node features (T, 5).

        [0] h2_prof_conflicts_ratio
        [1] h3_room_conflicts_ratio
        [2] load_ratio
        [3] free_rooms_ratio — fraction of rooms free at this timeslot
        [4] saturation       — total courses / total rooms at this timeslot
        """
        ds = self.ds
        sc = self.sched
        P, R, C = len(ds.professors), len(ds.rooms), len(ds.courses)

        h2_prof = (self.prof_ts > 1).astype(np.float32).sum(axis=0).to_numpy() / P
        h3_room = (self.room_ts > 1).astype(np.float32).sum(axis=0).to_numpy() / R

        load_ratio = sc['timeslot'].astype(int).value_counts().reindex(range(1, 26),
                                                                       fill_value=0).sort_index().to_numpy() / C

        free_rooms = (self._room_occupied == 0).sum(axis=0) / R

        courses_per_ts = self._room_occupied.sum(axis=0)
        saturation = courses_per_ts / max(R, 1)

        return np.stack([
            h2_prof,  # [0]
            h3_room,  # [1]
            load_ratio,  # [2]
            free_rooms,  # [3]
            saturation,  # [4]
        ], axis=1).astype(np.float32)

    # above fixed

    def _build_program_features(self) -> np.ndarray:
        """Program node features (G, 4).

        Programs participate in S2 (overlap).

        [0] s2_total_overlap_norm  — total overlap across all timeslots
        [1] s2_any_overlap         — binary flag
        [2] s2_worst_slot_norm     — worst single-slot overlap
        [3] num_courses_norm       — |C_g| / max|C_g|
        """

        # TODO: remove [3], add [4] number of courses in this program
        ds = self.ds
        sc = self.sched
        G = sc['dl'].unique().tolist()

        total = self.s2_overlap.sum(axis=1).to_numpy()
        max_total = max(total.max(), 1.0)
        total_norm = total / max_total

        any_overlap = (total > 0).astype(np.float32)

        worst = self.s2_overlap.max(axis=1)
        max_worst = max(worst.max(), 1.0)
        worst_norm = (worst / max_worst).to_numpy()

        counts = np.array([len(ds.C_g[g]) for g in G], dtype=np.float32)
        max_counts = max(counts.max(), 1.0)
        counts_norm = counts / max_counts

        return np.stack([
            total_norm,  # [0]
            any_overlap,  # [1]
            worst_norm,  # [2]
            counts_norm,  # [3]
        ], axis=1).astype(np.float32)

    def build_graph(self):
        self._check_constraints()
        self.data = HeteroData()

        course_to_idx = {course: i for i, course in enumerate(self.ds.course_list)}
        prof_to_idx = {prof: i for i, prof in enumerate(self.ds.professors['P-Index'].drop_duplicates())}
        room_to_idx = {room: i for i, room in enumerate(self.ds.rooms['Host Key'].drop_duplicates())}
        program_to_idx = {program: i for i, program in enumerate(self.sched['dl'].unique().tolist())}
        timeslot_to_idx = {timeslot: i for i, timeslot in enumerate(list(self.ds.timeslots))}

        course_features = self._build_course_features()
        professor_features = self._build_professor_features()
        room_features = self._build_room_features()
        program_features = self._build_program_features()
        timeslot_features = self._build_timeslot_features()

        self.data['course'].x = torch.tensor(course_features, dtype=torch.float)
        self.data['professor'].x = torch.tensor(professor_features, dtype=torch.float)
        self.data['Room'].x = torch.tensor(room_features, dtype=torch.float)
        self.data['program'].x = torch.tensor(program_features, dtype=torch.float)
        self.data['timeslot'].x = torch.tensor(timeslot_features, dtype=torch.float)

        course_professor = list(self.sched[['Index', 'P-Index']].itertuples(index=False, name=None))
        course_room = list(self.sched[['Index', 'Room']].itertuples(index=False, name=None))
        course_timeslot = list(self.sched[['Index', 'timeslot']].itertuples(index=False, name=None))
        course_program = list(self.sched[['Index', 'dl']].itertuples(index=False, name=None))

        self.data['course', 'taught_by', 'professor'].edge_index = make_edge_index(course_professor, course_to_idx,
                                                                                   prof_to_idx)
        self.data['course', 'take_place_in', 'Room'].edge_index = make_edge_index(course_room, course_to_idx,
                                                                                  room_to_idx)
        self.data['course', 'belong_to', 'program'].edge_index = make_edge_index(course_program, course_to_idx,
                                                                                 program_to_idx)
        self.data['course', 'when', 'timeslot'].edge_index = make_edge_index(course_timeslot, course_to_idx,
                                                                             timeslot_to_idx)

        # Validation
        node_counts = {
            'course': self.data['course'].x.size(0),
            'professor': self.data['professor'].x.size(0),
            'Room': self.data['Room'].x.size(0),
            'program': self.data['program'].x.size(0),
            'timeslot': self.data['timeslot'].x.size(0),
        }
        edge_checks = [
            (('course', 'taught_by', 'professor'), 'course', 'professor'),
            (('course', 'take_place_in', 'Room'), 'course', 'Room'),
            (('course', 'belong_to', 'program'), 'course', 'program'),
            (('course', 'when', 'timeslot'), 'course', 'timeslot'),
        ]
        for edge_type, src_type, dst_type in edge_checks:
            ei = self.data[edge_type].edge_index
            src_max = ei[0].max().item() if ei.size(1) > 0 else -1
            dst_max = ei[1].max().item() if ei.size(1) > 0 else -1
            assert src_max < node_counts[src_type], (
                f"Edge {edge_type}: source index {src_max} >= "
                f"num {src_type} nodes {node_counts[src_type]}"
            )
            assert dst_max < node_counts[dst_type], (
                f"Edge {edge_type}: target index {dst_max} >= "
                f"num {dst_type} nodes {node_counts[dst_type]}"
            )

        self.data = T.ToUndirected()(self.data)

        return self.data


def make_edge_index(edge_pairs, source, destination):
    src = []
    dst = []
    for src_id, dst_id in edge_pairs:
        src_idx = source[src_id]
        dst_idx = destination[dst_id]
        src.append(src_idx)
        dst.append(dst_idx)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


class HeteroGraphSAGE(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()

        node_types, edge_types = metadata

        self.convs = nn.ModuleList()

        for _ in range(2):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((-1, -1), hidden_channels)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin = nn.ModuleDict({
            node_type: Linear(hidden_channels, out_channels)
            for node_type in node_types
        })

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: F.relu(x)
                for node_type, x in x_dict.items()
            }

        out_dict = {
            node_type: self.lin[node_type](x)
            for node_type, x in x_dict.items()
        }

        return out_dict
