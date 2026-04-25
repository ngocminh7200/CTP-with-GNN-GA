import networkx as nx
import numpy as np
from Instance import Instance

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
        self.prof_ts = np.zeros((P, T), dtype=np.int32)
        for c in range(C):
            p = int(ds.course_professor[c])
            t = int(sc.assigned_timeslot[c])
            self.prof_ts[p, t] += 1

        # Per-course: does this course cause a professor conflict?
        self.h2_course = np.zeros(C, dtype=np.float32)
        for c in range(C):
            p = int(ds.course_professor[c])
            t = int(sc.assigned_timeslot[c])
            if self.prof_ts[p, t] > 1:
                self.h2_course[c] = 1.0

        # --- H3: Room conflict ---
        # room_ts[r, t] = number of courses assigned to room r at timeslot t
        self.room_ts = np.zeros((R, T), dtype=np.int32)
        for c in range(C):
            r = int(sc.assigned_room[c])
            t = int(sc.assigned_timeslot[c])
            self.room_ts[r, t] += 1

        self.h3_course = np.zeros(C, dtype=np.float32)
        for c in range(C):
            r = int(sc.assigned_room[c])
            t = int(sc.assigned_timeslot[c])
            if self.room_ts[r, t] > 1:
                self.h3_course[c] = 1.0

        # --- H4: Room capacity ---
        # Binary: is s_c > c_r for the assigned room?
        self.h4_course = np.zeros(C, dtype=np.float32)
        self.h4_overflow = np.zeros(C, dtype=np.float32)
        for c in range(C):
            r = int(sc.assigned_room[c])
            overflow = ds.enrollment[c] - ds.room_capacity[r]
            if overflow > 0:
                self.h4_course[c] = 1.0
                self.h4_overflow[c] = overflow

        max_overflow = max(self.h4_overflow.max(), 1.0)
        self.h4_overflow_norm = self.h4_overflow / max_overflow

        # --- S1: Global timeslot crowding ---
        # courses_at_t[t] = total courses at timeslot t
        self.courses_at_t = np.zeros(T, dtype=np.float32)
        for c in range(C):
            self.courses_at_t[int(sc.assigned_timeslot[c])] += 1.0

        # s^time_t = max(0, courses_at_t - |R|)
        self.s1_excess = np.maximum(0.0, self.courses_at_t - R)
        max_s1 = max(self.s1_excess.max(), 1.0)
        self.s1_norm = self.s1_excess / max_s1

        # --- S2: Program overlap ---
        # prog_ts[g, t] = courses from program g at timeslot t
        self.prog_ts = np.zeros((G, T), dtype=np.int32)
        for g in range(G):
            for c in ds.prog_courses[g]:
                t = int(sc.assigned_timeslot[c])
                self.prog_ts[g, t] += 1

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
        C = ds.num_courses

        enrollment_norm = ds.enrollment / (ds.enrollment.max() + 1e-8)

        room_fit = np.zeros(C, dtype=np.float32)
        for c in range(C):
            r = int(sc.assigned_room[c])
            room_fit[c] = ds.room_capacity[r] / (ds.enrollment[c] + 1e-8)
        room_fit = np.clip(room_fit, 0.0, 3.0)

        flexibility = np.array([len(ds.eligible_rooms[c]) / ds.num_rooms
                                for c in range(C)], dtype=np.float32)

        prog_counts = np.array([len(ds.course_programs[c]) for c in range(C)], dtype=np.float32)
        prog_counts_norm = prog_counts / (prog_counts.max() + 1e-8)

        # S2 exposure: how much overlap does this course contribute to?
        s2_exposure = np.zeros(C, dtype=np.float32)
        for c in range(C):
            t = int(sc.assigned_timeslot[c])
            for g in ds.course_programs[c]:
                s2_exposure[c] += self.s2_overlap[g, t]
        max_exp = max(s2_exposure.max(), 1.0)
        s2_exposure /= max_exp

        return np.stack([
            self.h2_course,  # [0]
            self.h3_course,  # [1]
            self.h4_course,  # [2]
            self.h4_overflow_norm,  # [3]
            enrollment_norm,  # [4]
            room_fit,  # [5]
            flexibility,  # [6]
            prog_counts_norm,  # [7]
            s2_exposure,  # [8]
        ], axis=1).astype(np.float32)

    def _build_professor_features(self) -> np.ndarray:
        """Professor node features (P, 4).

        Professors participate in H2 (conflict).

        [0] h2_any_conflict      — binary: any timeslot conflict?
        [1] h2_conflict_ratio    — fraction of timeslots with conflicts
        [2] num_courses_norm     — |C_p| / max|C_p|
        [3] teaching_density     — actual slots used / T (how spread out)
        """
        #TODO: question about [1], remove [2]
        ds = self.ds
        P, T = ds.num_professors, ds.num_timeslots

        conflicts = (self.prof_ts > 1).astype(np.float32)
        h2_any = (conflicts.sum(axis=1) > 0).astype(np.float32)
        h2_ratio = conflicts.sum(axis=1) / (T + 1e-8)

        course_counts = np.array([len(ds.prof_courses[p]) for p in range(P)], dtype=np.float32)
        course_counts_norm = course_counts / (course_counts.max() + 1e-8)

        active_slots = (self.prof_ts > 0).astype(np.float32).sum(axis=1)
        teaching_density = active_slots / (T + 1e-8)

        return np.stack([
            h2_any,  # [0]
            h2_ratio,  # [1]
            course_counts_norm,  # [2]
            teaching_density,  # [3]
        ], axis=1).astype(np.float32)

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
        ds = self.ds
        R, T, C = ds.num_rooms, ds.num_timeslots, ds.num_courses

        conflicts = (self.room_ts > 1).astype(np.float32)
        h3_any = (conflicts.sum(axis=1) > 0).astype(np.float32)
        h3_ratio = conflicts.sum(axis=1) / (T + 1e-8)

        occupancy = (self.room_ts > 0).astype(np.float32).sum(axis=1) / T
        cap_norm = ds.room_capacity / (ds.room_capacity.max() + 1e-8)

        # Average overflow for courses assigned to each room
        room_overflows = defaultdict(list)
        for c in range(C):
            r = int(self.sched.assigned_room[c])
            ov = max(0.0, ds.enrollment[c] - ds.room_capacity[r])
            room_overflows[r].append(ov)
        max_enrollment = ds.enrollment.max()
        avg_overflow = np.zeros(R, dtype=np.float32)
        for r in range(R):
            if room_overflows[r]:
                avg_overflow[r] = np.mean(room_overflows[r]) / (max_enrollment + 1e-8)

        return np.stack([
            h3_any,  # [0]
            h3_ratio,  # [1]
            occupancy,  # [2]
            cap_norm,  # [3]
            avg_overflow,  # [4]
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

        ds = self.ds
        T, P, R, C = ds.num_timeslots, ds.num_professors, ds.num_rooms, ds.num_courses
        G = ds.num_programs

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
