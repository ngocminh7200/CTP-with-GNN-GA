import networkx as nx
import numpy as np
from Instance import Instance
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv,SAGEConv,Linear


class Graph:

    def __init__(self, instance: Instance, schedule):
        self.G = nx.Graph()
        self.ds = instance
        self.sched = schedule

    def _check_constraints(self):
        """Compute all constraint violations for the current schedule.
        Results are cached as instance attributes for use by feature builders.
        """
        ds, sc = self.ds, self.sched
        C, R, T, P, G = self.ds.get_sizes()

        # --- H2: Professor conflict ---
        # prof_ts[p, t] = number of courses professor p teaches at timeslot t

        self.prof_ts = pd.crosstab(sc['professor'], sc['timeslot']).reindex(columns=ds.timeslots, fill_value=0)



        # Per-course: does this course cause a professor conflict?
        self.h2_course = np.zeros(C, dtype=np.float32)
        for c in range(C):
            prf=sc.iloc[c]['professor']
            ts=sc.iloc[c]['timeslot']
            if self.prof_ts[ts][prf]>1:
                self.h2_course[c]=1

        # --- H3: Room conflict ---
        # room_ts[r, t] = number of courses assigned to room r at timeslot t
        self.room_ts=pd.crosstab(sc['classroom index'], sc['timeslot']).reindex(columns=ds.timeslots, fill_value=0)

        self.h3_course = np.zeros(C, dtype=np.float32)
        for c in range(C):
            rm=sc.iloc[c]['classroom index']
            ts=sc.iloc[c]['timeslot']
            if self.room_ts[ts][rm] > 1:
                self.h3_course[c] = 1.0

        # --- H4: Room capacity ---
        # Binary: is s_c > c_r for the assigned room?
        self.h4_course = np.zeros(C, dtype=np.float32)
        self.h4_overflow = np.zeros(C, dtype=np.float32)
        for c in range(C):
            overflow = sc.iloc[c]['numstu'] -sc.iloc[c]['capacity']
            if overflow > 0:
                self.h4_course[c] = 1.0
                self.h4_overflow[c] = overflow

        max_overflow = max(self.h4_overflow.max(), 1.0)
        self.h4_overflow_norm = self.h4_overflow / max_overflow

        # --- S1: Global timeslot crowding ---
        # courses_at_t[t] = total courses at timeslot t
        self.courses_at_t=sc.groupby('timeslot').size()


        # --- S2: Program overlap ---
        # prog_ts[g, t] = courses from program g at timeslot t
        self.prog_ts=pd.crosstab(sc['dl'], sc['timeslot']).reindex(columns=ds.timeslots, fill_value=0)

        # s^overlap_{g,t} = max(0, prog_ts - 1)
        self.s2_overlap = np.maximum(0, self.prog_ts - 1).astype(np.float32)

    def _build_course_features(self) -> np.ndarray:
        """Course node features (C, 9).

        Courses are the central entity — the decision variable x_{c,t,r}
        lives here. They participate in every constraint.

        [0] h2_violation       — professor conflict flag
        [1] h3_violation       — room conflict flag
        [2] h4_violation       — capacity violation flag
        [3] h4_overflow_norm   — normalized capacity overflow degree
        [4] enrollment_norm    — s_c / max(s_c)
        [5] room_fit           — c_r / s_c for assigned room (>1 = fits)
        [6] flexibility        — fraction of rooms this course can use (from H4)
        [7] num_programs_norm  — how many programs this course is in / max
        [8] s2_exposure        — sum of program overlaps at this course's timeslot
                                 across all programs it belongs to, normalized
        """
        #TODO: remove [4], [6], adjust [8] (cuz for us, a course only belong to 1 prog)
        ds, sc = self.ds, self.sched
        C = len(ds.course_list)


        room_fit=(sc['capacity']/sc['numstu']).to_numpy()
        room_fit = np.clip(room_fit, 0.0, 3.0)

        flexibility = np.array([(ds.rooms['Capacity']>=ds.courses.iloc[c]['NUMSTU']).mean()
                                for c in range(C)], dtype=np.float32)


        return np.stack([
            self.h2_course,  # [0]
            self.h3_course,  # [1]
            self.h4_course,  # [2]
            self.h4_overflow_norm,  # [3]
            room_fit,  # [5]
            flexibility,  # [6]
        ], axis=1).astype(np.float32)
# above has been fixed

    def _build_professor_features(self) -> np.ndarray:
        """Professor node features (P, 4).

        Professors participate in H2 (conflict).

        [0] h2_any_conflict      — binary: any timeslot conflict?
        [1] h2_conflict_ratio    — fraction of timeslots with conflicts
        [2] num_courses_norm     — |C_p| / max|C_p|
        [3] teaching_density     — actual slots used / T (how spread out)
        """
        #TODO: question about [1], remove [2]
        # remove[1], what is this?
        ds = self.ds
        T = len(ds.timeslots)

        conflicts = (self.prof_ts > 1).astype(np.float32)
        h2_any = (conflicts.sum(axis=1).to_numpy() > 0).astype(np.float32)

        course_counts = np.array([len(ds.C_p[p]) for p in ds.professors], dtype=np.float32)


        active_slots = (self.prof_ts > 0).astype(np.float32).sum(axis=1).to_numpy()
        teaching_density=active_slots/T

        return np.stack([
            h2_any,  # [0]
            course_counts,  # [2]??
            teaching_density,  # [3]
        ], axis=1).astype(np.float32)
    
# above fixed

    def _build_room_features(self) -> np.ndarray:
        """Room node features (R, 5).

        Rooms participate in H3 (conflict) and H4 (capacity).

        [0] h3_any_conflict      — binary: any double-booking?
        [1] h3_conflict_ratio    — fraction of timeslots double-booked
        [2] occupancy_rate       — fraction of timeslots in use
        [3] capacity_norm        — c_r / max(c_r)
        [4] avg_overflow_norm    — average capacity overflow for assigned courses
        """
        #TODO: remove [3]
        # remove [4] as well. goodbye~
        ds = self.ds
        T = len(ds.timeslots)

        conflicts = (self.room_ts > 1).astype(np.float32)
        h3_any = (conflicts.sum(axis=1).to_numpy() > 0).astype(np.float32)
        h3_ratio = conflicts.sum(axis=1)/T

        occupancy = (self.room_ts > 0).astype(np.float32).sum(axis=1).to_numpy()/T

        return np.stack([
            h3_any,  # [0]
            h3_ratio,  # [1]
            occupancy,  # [2]
        ], axis=1).astype(np.float32)

    def _build_timeslot_features(self) -> np.ndarray:
        """Timeslot node features (T, 6).

        Timeslots aggregate conflict info across H2, H3, S1, S2.

        [0] h2_prof_conflicts_ratio  — fraction of professors with conflicts at t
        [1] h3_room_conflicts_ratio  — fraction of rooms double-booked at t
        [2] s1_crowding_norm         — global timeslot crowding penalty, normalized
        [3] load_ratio               — courses_at_t / C
        [4] day_index_norm           — which day (0–1)
        [5] slot_in_day_norm         — position within the day (0–1)
        """
        #TODO: 
            # removed [4] and [5], [2] as well

        ds=self.ds
        sc=self.sched
        P,R,C=len(ds.professors),len(ds.rooms),len(ds.courses)


        h2_prof = (self.prof_ts > 1).astype(np.float32).sum(axis=0).to_numpy()/P
        h3_room = (self.room_ts > 1).astype(np.float32).sum(axis=0).to_numpy()/R

        load_ratio = sc['timeslot'].astype(int).value_counts().reindex(range(1, 26), fill_value=0).sort_index().to_numpy()/ C


        return np.stack([
            h2_prof,  # [0]
            h3_room,  # [1]
            load_ratio,  # [3]

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

        #TODO: remove [3], add [4] number of courses in this program
        ds=self.ds
        sc=self.sched
        G=sc['dl'].unique().tolist()

        total = self.s2_overlap.sum(axis=1).to_numpy()
        max_total = max(total.max(), 1.0)
        total_norm = total / max_total

        any_overlap = (total > 0).astype(np.float32)

        worst = self.s2_overlap.max(axis=1)
        max_worst = max(worst.max(), 1.0)
        worst_norm = (worst/max_worst).to_numpy()

        counts = np.array([len(ds.C_g[g]) for g in G], dtype=np.float32)
        counts_norm = counts/counts.max()

        return np.stack([
            total_norm,  # [0]
            any_overlap,  # [1]
            worst_norm,  # [2]
            counts_norm,  # [3]
        ], axis=1).astype(np.float32)


    def build_graph(self):
        self._check_constraints()
        self.data=HeteroData()
        
        course_to_idx={course:i for i, course in enumerate(self.ds.course_list)}
        prof_to_idx={prof:i for i,prof in enumerate(self.ds.professors)}
        room_to_idx={room:i for i,room in enumerate(self.ds.rooms['R-Index'].tolist())}
        program_to_idx={program:i for i, program in enumerate(self.sched['dl'].unique().tolist())}
        timeslot_to_idx={timeslot:i for i, timeslot in enumerate(list(self.ds.timeslots))}

        
        course_features=self._build_course_features()
        professor_features=self._build_professor_features()
        room_features=self._build_room_features()
        program_features=self._build_program_features()
        timeslot_features=self._build_timeslot_features()
        print(timeslot_features)
        
        self.data['course'].x=torch.tensor(course_features,dtype=torch.float)
        self.data['professor'].x=torch.tensor(professor_features,dtype=torch.float)
        self.data['room'].x=torch.tensor(room_features,dtype=torch.float)
        self.data['program'].x=torch.tensor(program_features,dtype=torch.float)
        self.data['timeslot'].x=torch.tensor(timeslot_features,dtype=torch.float)
        
        course_professor=list(self.sched[['course index','professor']].itertuples(index=False,name=None))
        course_room=list(self.sched[['course index','classroom index']].itertuples(index=False,name=None))
        course_timeslot=list(self.sched[['course index','timeslot']].itertuples(index=False,name=None))
        course_program=list(self.sched[['course index','dl']].itertuples(index=False,name=None))
        
        self.data['course','taught_by','professor'].edge_index=make_edge_index(course_professor, course_to_idx, prof_to_idx)
        self.data['course','take_place_in','room'].edge_index=make_edge_index(course_room, course_to_idx, room_to_idx)
        self.data['course','belong_to','program'].edge_index=make_edge_index(course_program, course_to_idx, program_to_idx)
        self.data['course','when','timeslot'].edge_index=make_edge_index(course_timeslot, course_to_idx, timeslot_to_idx)
        
        self.data=T.ToUndirected()(self.data)
    
    def build_model(self):
        self.build_graph()
        self.model=HeteroGraphSAGE(64, 32, self.data.metadata())
        self.out=self.model(self.data.x_dict,self.data.edge_index_dict)
        


def make_edge_index(edge_pairs,source,destination):
    src=[]
    dst=[]
    for src_id,dst_id in edge_pairs:
        src_idx=source[src_id]
        dst_idx=destination[dst_id]
        src.append(src_idx)
        dst.append(dst_idx)
    edge_index=torch.tensor([src,dst],dtype=torch.long)
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
        
        
        
        
