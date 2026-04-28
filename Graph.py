import networkx as nx
import numpy as np
from Instance import Instance
import pandas as pd

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

        self.prof_ts=pd.crosstab(sc['professor'],sc['timeslot'])


        # Per-course: does this course cause a professor conflict?
        self.h2_course = np.zeros(C, dtype=np.float32)
        for c in range(C):
            prf=sc.iloc[c]['professor']
            ts=sc.iloc[c]['timeslot']
            if self.prof_ts[ts][prf]>1:
                self.h2_course[c]=1

        # --- H3: Room conflict ---
        # room_ts[r, t] = number of courses assigned to room r at timeslot t
        self.room_ts=pd.crosstab(sc['classroom index'], sc['timeslot'])

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

        # s^time_t = max(0, courses_at_t - |R|)
        self.s1_excess = np.maximum(0.0, self.courses_at_t - R)
        max_s1 = max(self.s1_excess.max(), 1.0)
        self.s1_norm = self.s1_excess / max_s1

        # --- S2: Program overlap ---
        # prog_ts[g, t] = courses from program g at timeslot t
        self.prog_ts=pd.crosstab(sc['dl'], sc['timeslot'])

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
        ds, sc = self.ds, self.sched
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
        R, T, C = len(ds.rooms), len(ds.timeslots), len(ds.courses)

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

        ds=self.ds
        sc=self.sched
        T, P, R, C = len(ds.timeslots), len(ds.professors), len(ds.rooms), len(ds.courses)
        G = sc['dl'].unique().tolist()

        h2_prof = (self.prof_ts > 1).astype(np.float32).sum(axis=0) / (P + 1e-8)
        h3_room = (self.room_ts > 1).astype(np.float32).sum(axis=0) / (R + 1e-8)

        load_ratio = self.courses_at_t / (C + 1e-8)

        day_idx = np.array([t // ds.slots_per_day for t in range(T)], dtype=np.float32)
        day_norm = day_idx / (ds.num_days + 1e-8)
        slot_in_day = np.array([t % ds.slots_per_day for t in range(T)], dtype=np.float32)
        slot_norm = slot_in_day / (ds.slots_per_day + 1e-8)

        return np.stack([
            h2_prof,  # [0]
            h3_room,  # [1]
            self.s1_norm,  # [2]
            load_ratio,  # [3]
            day_norm,  # [4]
            slot_norm,  # [5]
        ], axis=1).astype(np.float32)

    def _build_program_features(self) -> np.ndarray:
        """Program node features (G, 4).

        Programs participate in S2 (overlap).

        [0] s2_total_overlap_norm  — total overlap across all timeslots
        [1] s2_any_overlap         — binary flag
        [2] s2_worst_slot_norm     — worst single-slot overlap
        [3] num_courses_norm       — |C_g| / max|C_g|
        """

        #TODO: remove [3], add [4] number of courses in this program
        ds = self.ds
        G = ds.num_programs

        total = self.s2_overlap.sum(axis=1)
        max_total = max(total.max(), 1.0)
        total_norm = total / max_total

        any_overlap = (total > 0).astype(np.float32)

        worst = self.s2_overlap.max(axis=1)
        max_worst = max(worst.max(), 1.0)
        worst_norm = worst / max_worst

        counts = np.array([len(ds.prog_courses[g]) for g in range(G)], dtype=np.float32)
        counts_norm = counts / (counts.max() + 1e-8)

        return np.stack([
            total_norm,  # [0]
            any_overlap,  # [1]
            worst_norm,  # [2]
            counts_norm,  # [3]
        ], axis=1).astype(np.float32)

    def _build_features(self):
        """Compute and store all node features."""
        ds = self.ds

        course_feats = self._build_course_features()
        prof_feats = self._build_professor_features()
        room_feats = self._build_room_features()
        ts_feats = self._build_timeslot_features()
        prog_feats = self._build_program_features()

        # Store individually
        for c in range(ds.num_courses):
            self._features[('course', c)] = course_feats[c]
        for p in range(ds.num_professors):
            self._features[('professor', p)] = prof_feats[p]
        for r in range(ds.num_rooms):
            self._features[('room', r)] = room_feats[r]
        for t in range(ds.num_timeslots):
            self._features[('timeslot', t)] = ts_feats[t]
        for g in range(ds.num_programs):
            self._features[('program', g)] = prog_feats[g]

        # Also cache the matrices for fast retrieval
        self._feat_matrices = {
            'course': course_feats,
            'professor': prof_feats,
            'room': room_feats,
            'timeslot': ts_feats,
            'program': prog_feats,
        }

    def addEdge(self, u, v):
        self.G.add_edge(u, v)

    def printGraph(self):
        print(self.G)
