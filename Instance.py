import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def slot_label(t, slots_per_day=5):
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    day_idx = (t - 1) // slots_per_day
    slot_idx = (t - 1) % slots_per_day + 1
    start_hour = 8 + (slot_idx - 1) * 2  # first slot starts at 08:00
    return f"{day_names[day_idx]} {start_hour:02d}:00-{start_hour + 2:02d}:00"

class Instance:
    def __init__(self, courses, rooms, timeslots, professors):
        self.courses = courses
        self.rooms = rooms
        self.timeslots = range(1,timeslots+1)
        self.professors = professors

        #TODO: remove duplicates
        self.course_list = courses['Index'].tolist()
        self.room_list = rooms['Host Key'].tolist()

        self.C_p = courses.groupby('P-Index')['Index'].apply(list).to_dict()
        self.C_g = courses.groupby('dl')['Index'].apply(list).to_dict()
        self.demand = courses.set_index('Index')['NUMSTU'].to_dict()
        self.capacity = rooms.set_index('Host Key')['Capacity'].to_dict()

        #self.alpha = 0.1
        self.beta = 0.1
        self.gamma = 0.01

    def generate_Gurobi_model(self):

        self.model = gp.Model("CTP")

        # ── Decision variables ──

        # x[c,t,r] — primary binary assignment
        self.x = self.model.addVars(
            self.course_list, self.timeslots, self.room_list,
            vtype=GRB.BINARY, name="x"
        )

        # s_overlap[g,t] — program overlap slack
        self.s_overlap = self.model.addVars(
            list(self.C_g.keys()), self.timeslots,
            vtype=GRB.CONTINUOUS, lb=0.0, name="s_overlap"
        )

        # s_overlap_global[t] — overall overlap slack
        self.s_overlap_global = self.model.addVars(
            self.timeslots,
            vtype=GRB.CONTINUOUS, lb=0.0, name="s_overlap_global"
        )

        # s_cap[c,t,r] — capacity overflow slack
        # s_cap = model.addVars(
        #     courses, timeslots, room_list,
        #     vtype=GRB.CONTINUOUS, lb=0.0, name="s_cap"
        # )

        self.model.setObjective(
            # self.alpha * s_cap.sum()
            # +
            self.beta * self.s_overlap.sum()
            + self.gamma * self.s_overlap_global.sum(),
            GRB.MINIMIZE,
        )

        # ──────────────────────────────────────────────
        # 4.  HARD CONSTRAINTS
        # ──────────────────────────────────────────────

        # H1 — Course assignment: each course assigned to exactly one (t, r) pair
        for c in self.course_list:
            self.model.addConstr(
                gp.quicksum(self.x[c, t, r] for t in self.timeslots for r in self.room_list) == 1,
                name=f"H1_assign_{c}",
            )

        # H2 — Room conflict: at most one course per room per timeslot
        for r in self.room_list:
            for t in self.timeslots:
                self.model.addConstr(
                    gp.quicksum(self.x[c, t, r] for c in self.course_list) <= 1,
                    name=f"H2_room_{r}_t{t}",
                )

        # H3 — Professor conflict: at most one course per professor per timeslot
        for p, p_courses in self.C_p.items():
            for t in self.timeslots:
                self.model.addConstr(
                    gp.quicksum(self.x[c, t, r] for c in p_courses for r in self.room_list) <= 1,
                    name=f"H3_prof_{p}_t{t}",
                )

        # ──────────────────────────────────────────────
        # 5.  SOFT CONSTRAINTS
        # ──────────────────────────────────────────────

        # S1 — Room capacity:  s_cap[c,t,r] >= (d_c - κ_r) * x[c,t,r]
        for c in self.course_list:
            for t in self.timeslots:
                for r in self.room_list:
                    overflow = self.demand[c] - self.capacity[r]
                    if overflow > 0:
                        self.model.addConstr(
                            # s_cap[c, t, r] >= overflow * x[c, t, r],
                            0 >= overflow * self.x[c, t, r],
                            name=f"S1_cap_{c}_{r}_t{t}",
                        )
                    # When overflow <= 0, s_cap >= 0 is already satisfied by lb=0

        # S2 — Program course overlap:
        #   Σ_{c ∈ C_g} Σ_r x[c,t,r] - 1  <=  s_overlap[g,t]
        for g, g_courses in self.C_g.items():
            for t in self.timeslots:
                self.model.addConstr(
                    gp.quicksum(self.x[c, t, r] for c in g_courses for r in self.room_list)
                    - 1
                    <= self.s_overlap[g, t],
                    name=f"S2_overlap_{g}_t{t}",
                )

        # S3 - Global time overlap:
        for t in self.timeslots:
            self.model.addConstr(
                gp.quicksum(self.x[c, t, r] for c in self.course_list for r in self.room_list)
                - len(self.room_list)
                <= self.s_overlap_global[t],
                name=f"S3_global_overlap_t{t}",
            )

    def solve_Gurobi(self):
        self.model.Params.TimeLimit = 300  # 5 min time limit
        self.model.Params.MIPGap = 0.01  # 1% optimality gap
        self.model.optimize()


    def print_Gurobi_solution(self):
        if self.model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            print("\n" + "=" * 65)
            print("  OPTIMAL TIMETABLE")
            print("=" * 65)
            print(f"  Objective value: {self.model.ObjVal:.2f}")
            # print(f"  Total capacity penalty:  "
            #       f"{alpha * sum(s_cap[c,t,r].X for c in courses for t in timeslots for r in room_list):.2f}")
            print(f"  Total overlap penalty:   "
                  f"{self.beta * sum(self.s_overlap[g, t].X for g in self.C_g.keys() for t in self.timeslots):.2f}")
            # print(f"  Total workload penalty:  "
            #       f"{self.gamma * sum(s_wl[p].X for p in professors):.2f}")
            # print("-" * 65)
            print(f"  {'Course':<10} {'Timeslot':<22} {'Room':<6} "
                  f"{'Demand':<8} {'Cap':<6} {'Overflow'}")
            print("-" * 65)

            for c in self.course_list:
                for t in self.timeslots:
                    for r in self.room_list:
                        if self.x[c, t, r].X > 0.5:
                            overflow = max(0, self.demand[c] - self.capacity[r])
                            flag = " !!" if overflow > 0 else ""
                            print(f"  {c:<10} {slot_label(t):<22} {r:<6} "
                                  f"{self.demand[c]:<8} {self.capacity[r]:<6} {overflow}{flag}")

            # Program overlap report
            print("\n" + "-" * 65)
            print("  PROGRAM OVERLAP VIOLATIONS")
            print("-" * 65)
            any_overlap = False
            for g in self.C_g.keys():
                for t in self.timeslots:
                    val = self.s_overlap[g, t].X
                    if val > 0.5:
                        any_overlap = True
                        print(f"  {g:<18} {slot_label(t):<22} "
                              f"excess courses: {val:.0f}")
            if not any_overlap:
                print("  None - all programs are conflict-free!")

            # Workload report
            # print("\n" + "-" * 65)
            # print("  INSTRUCTOR WORKLOAD")
            # print("-" * 65)
            # for p in professors:
            #     assigned = sum(
            #         1 for c in C_p[p] for t in timeslots for r in room_list
            #         if x[c, t, r].X > 0.5
            #     )
            #     excess = s_wl[p].X
            #     flag = " !!" if excess > 0.5 else ""
            #     print(f"  {p:<12} assigned: {assigned}/{W[p]} slots{flag}")
            #
            # print("=" * 65)

        elif self.model.Status == GRB.INFEASIBLE:
            print("\nModel is infeasible. Computing IIS...")
            self.model.computeIIS()
            self.model.write("ctp_iis.ilp")
            print("IIS written to ctp_iis.ilp - inspect to identify conflicting constraints.")

        else:
            print(f"\nOptimization ended with status {self.model.Status}")


    

    def get_sizes(self):
        return len(self.course_list), len(self.room_list), len(self.timeslots), len(self.C_p), len(self.C_g)
    
    
    def df_solution(self):
        if self.model.status==GRB.INFEASIBLE:
            return None
        df=[]
        for c in self.course_list:
            for t in self.timeslots:
                for r in self.room_list:
                    if self.x[c,t,r].X>0.5:
                        df.append({'course index':c,'timeslot':t,'room':r})
        return pd.DataFrame(df)

    # def export_solution(self):
    #     if self.model.Status == GRB.INFEASIBLE:
    #         self.model.write("")
