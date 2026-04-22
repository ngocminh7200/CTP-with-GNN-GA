"""
College Timetabling Problem (CTP) — Gurobi MILP Implementation
================================================================
Based on the GNN–GA hybrid framework (Cornei & Breabăn, 2026).

Decision variable:  x[c,t,r] ∈ {0,1}
    1 if course c is assigned to timeslot t in room r; 0 otherwise.

Timeslots are flat indices: 5 slots/day × number of days.
Each slot is a fixed 2-hour block (uniform duration for all courses).
"""

import gurobipy as gp
from gurobipy import GRB

# ──────────────────────────────────────────────
# 1.  SAMPLE DATA  (replace with your own)
# ──────────────────────────────────────────────

# Courses
courses = ["CS101", "CS201", "CS301", "MATH101", "MATH201",
           "PHY101", "PHY201", "ENG101", "ENG201", "STAT101"]

# Professors and the courses they teach
professors = {
    "Prof_A": ["CS101", "CS201"],
    "Prof_B": ["CS301", "STAT101"],
    "Prof_C": ["MATH101", "MATH201"],
    "Prof_D": ["PHY101", "PHY201"],
    "Prof_E": ["ENG101", "ENG201"],
}

# Maximum weekly teaching load per professor (number of 2-hr slots)
W = {
    "Prof_A": 4,
    "Prof_B": 3,
    "Prof_C": 4,
    "Prof_D": 3,
    "Prof_E": 4,
}

# Rooms and their capacities
rooms = {
    "R1": 40,
    "R2": 60,
    "R3": 100,
    "R4": 30,
}

# Student demand (enrollment) per course
demand = {
    "CS101": 90, "CS201": 55, "CS301": 35,
    "MATH101": 80, "MATH201": 45,
    "PHY101": 70, "PHY201": 30,
    "ENG101": 95, "ENG201": 50,
    "STAT101": 60,
}

# Academic programs and their course lists
programs = {
    "CS_Program":   ["CS101", "CS201", "CS301", "MATH101", "STAT101"],
    "Math_Program": ["MATH101", "MATH201", "STAT101", "CS101"],
    "Phys_Program": ["PHY101", "PHY201", "MATH101", "MATH201"],
    "Eng_Program":  ["ENG101", "ENG201", "MATH101"],
}

# Timeslots: 5 slots/day × 5 days = 25 weekly slots
num_days = 1
slots_per_day = 4
num_timeslots = num_days * slots_per_day
timeslots = list(range(1, num_timeslots + 1))

# Penalty weights (α, β, γ)
alpha = 1.0   # room capacity
beta  = 10.0  # program overlap  (primary objective → highest weight)
gamma = 2.0   # workload excess


# ──────────────────────────────────────────────
# 2.  DERIVED SETS
# ──────────────────────────────────────────────

room_list = list(rooms.keys())
kappa = rooms                        # κ_r  — room capacity dict
C_p = professors                     # C_p  — courses per professor
C_g = programs                       # C_g  — courses per program


# ──────────────────────────────────────────────
# 3.  MODEL
# ──────────────────────────────────────────────

model = gp.Model("CTP")

# ── 3a. Decision variables ──

# x[c,t,r] — primary binary assignment
x = model.addVars(
    courses, timeslots, room_list,
    vtype=GRB.BINARY, name="x"
)

# s_cap[c,t,r] — capacity overflow slack
# s_cap = model.addVars(
#     courses, timeslots, room_list,
#     vtype=GRB.CONTINUOUS, lb=0.0, name="s_cap"
# )

# s_overlap[g,t] — program overlap slack
s_overlap = model.addVars(
    list(programs.keys()), timeslots,
    vtype=GRB.CONTINUOUS, lb=0.0, name="s_overlap"
)

# s_wl[p] — workload excess slack
s_wl = model.addVars(
    list(professors.keys()),
    vtype=GRB.CONTINUOUS, lb=0.0, name="s_wl"
)


# ── 3b. Objective function ──
# min  α Σ s_cap  +  β Σ s_overlap  +  γ Σ s_wl

model.setObjective(
    # alpha * s_cap.sum()
    # +
    beta * s_overlap.sum()
    + gamma * s_wl.sum(),
    GRB.MINIMIZE,
)


# ──────────────────────────────────────────────
# 4.  HARD CONSTRAINTS
# ──────────────────────────────────────────────

# H1 — Course assignment: each course assigned to exactly one (t, r) pair
for c in courses:
    model.addConstr(
        gp.quicksum(x[c, t, r] for t in timeslots for r in room_list) == 1,
        name=f"H1_assign_{c}",
    )

# H2 — Room conflict: at most one course per room per timeslot
for r in room_list:
    for t in timeslots:
        model.addConstr(
            gp.quicksum(x[c, t, r] for c in courses) <= 1,
            name=f"H2_room_{r}_t{t}",
        )

# H3 — Professor conflict: at most one course per professor per timeslot
for p, p_courses in C_p.items():
    for t in timeslots:
        model.addConstr(
            gp.quicksum(x[c, t, r] for c in p_courses for r in room_list) <= 1,
            name=f"H3_prof_{p}_t{t}",
        )


# ──────────────────────────────────────────────
# 5.  SOFT CONSTRAINTS
# ──────────────────────────────────────────────

# S1 — Room capacity:  s_cap[c,t,r] >= (d_c - κ_r) * x[c,t,r]
for c in courses:
    for t in timeslots:
        for r in room_list:
            overflow = demand[c] - kappa[r]
            if overflow > 0:
                model.addConstr(
                    # s_cap[c, t, r] >= overflow * x[c, t, r],
                    0 >= overflow * x[c, t, r],
                    name=f"S1_cap_{c}_{r}_t{t}",
                )
            # When overflow <= 0, s_cap >= 0 is already satisfied by lb=0

# S2 — Program course overlap:
#   Σ_{c ∈ C_g} Σ_r x[c,t,r] - 1  <=  s_overlap[g,t]
for g, g_courses in C_g.items():
    for t in timeslots:
        model.addConstr(
            gp.quicksum(x[c, t, r] for c in g_courses for r in room_list)
            - 1
            <= s_overlap[g, t],
            name=f"S2_overlap_{g}_t{t}",
        )

# S3 — Instructor workload:
#   Σ_{c ∈ C_p} Σ_t Σ_r x[c,t,r]  -  W_p  <=  s_wl[p]
for p, p_courses in C_p.items():
    model.addConstr(
        gp.quicksum(
            x[c, t, r]
            for c in p_courses for t in timeslots for r in room_list
        )
        - W[p]
        <= s_wl[p],
        name=f"S3_workload_{p}",
    )


# ──────────────────────────────────────────────
# 6.  SOLVE
# ──────────────────────────────────────────────

model.Params.TimeLimit = 300        # 5 min time limit
model.Params.MIPGap = 0.01         # 1% optimality gap
model.optimize()


# ──────────────────────────────────────────────
# 7.  SOLUTION REPORT
# ──────────────────────────────────────────────

def slot_label(t):
    """Convert flat timeslot index to a human-readable day + slot string."""
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    day_idx = (t - 1) // slots_per_day
    slot_idx = (t - 1) % slots_per_day + 1
    start_hour = 8 + (slot_idx - 1) * 2  # first slot starts at 08:00
    return f"{day_names[day_idx]} {start_hour:02d}:00-{start_hour+2:02d}:00"


if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
    print("\n" + "=" * 65)
    print("  OPTIMAL TIMETABLE")
    print("=" * 65)
    print(f"  Objective value: {model.ObjVal:.2f}")
    # print(f"  Total capacity penalty:  "
    #       f"{alpha * sum(s_cap[c,t,r].X for c in courses for t in timeslots for r in room_list):.2f}")
    print(f"  Total overlap penalty:   "
          f"{beta * sum(s_overlap[g,t].X for g in programs for t in timeslots):.2f}")
    print(f"  Total workload penalty:  "
          f"{gamma * sum(s_wl[p].X for p in professors):.2f}")
    print("-" * 65)
    print(f"  {'Course':<10} {'Timeslot':<22} {'Room':<6} "
          f"{'Demand':<8} {'Cap':<6} {'Overflow'}")
    print("-" * 65)

    for c in courses:
        for t in timeslots:
            for r in room_list:
                if x[c, t, r].X > 0.5:
                    overflow = max(0, demand[c] - kappa[r])
                    flag = " !!" if overflow > 0 else ""
                    print(f"  {c:<10} {slot_label(t):<22} {r:<6} "
                          f"{demand[c]:<8} {kappa[r]:<6} {overflow}{flag}")

    # Program overlap report
    print("\n" + "-" * 65)
    print("  PROGRAM OVERLAP VIOLATIONS")
    print("-" * 65)
    any_overlap = False
    for g in programs:
        for t in timeslots:
            val = s_overlap[g, t].X
            if val > 0.5:
                any_overlap = True
                print(f"  {g:<18} {slot_label(t):<22} "
                      f"excess courses: {val:.0f}")
    if not any_overlap:
        print("  None - all programs are conflict-free!")

    # Workload report
    print("\n" + "-" * 65)
    print("  INSTRUCTOR WORKLOAD")
    print("-" * 65)
    for p in professors:
        assigned = sum(
            1 for c in C_p[p] for t in timeslots for r in room_list
            if x[c, t, r].X > 0.5
        )
        excess = s_wl[p].X
        flag = " !!" if excess > 0.5 else ""
        print(f"  {p:<12} assigned: {assigned}/{W[p]} slots{flag}")

    print("=" * 65)

elif model.Status == GRB.INFEASIBLE:
    print("\nModel is infeasible. Computing IIS...")
    model.computeIIS()
    model.write("ctp_iis.ilp")
    print("IIS written to ctp_iis.ilp - inspect to identify conflicting constraints.")

else:
    print(f"\nOptimization ended with status {model.Status}")