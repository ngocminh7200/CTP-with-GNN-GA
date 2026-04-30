"""
College Timetabling Problem (CTP) — Gurobi MILP Implementation

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

import random

random.seed(42)  # reproducibility

# ── 1a. Departments & course generation ──
# 3 departments, each with a mix of 100/200/300/400-level courses.
# Total: ~67 courses per dept ≈ 200 courses.

departments = {
    "CS": {"prefix": "CS", "n_courses": 70},
    "MATH": {"prefix": "MATH", "n_courses": 65},
    "PHYS": {"prefix": "PHYS", "n_courses": 65},
}

courses = []
demand = {}
dept_courses = {d: [] for d in departments}  # track courses per department

course_levels = [100, 200, 300, 400]
level_demand_range = {  # enrollment ranges by level
    100: (60, 150),  # intro courses — large
    200: (40, 100),  # intermediate
    300: (20, 60),  # upper-division
    400: (10, 35),  # senior/elective — small
}

for dept, info in departments.items():
    prefix = info["prefix"]
    n = info["n_courses"]
    # distribute courses roughly across levels
    per_level = n // len(course_levels)
    remainder = n % len(course_levels)
    idx = 1
    for li, level in enumerate(course_levels):
        count = per_level + (1 if li < remainder else 0)
        lo, hi = level_demand_range[level]
        for _ in range(count):
            cname = f"{prefix}{level + idx}"
            courses.append(cname)
            demand[cname] = random.randint(lo, hi)
            dept_courses[dept].append(cname)
            idx += 1

print(f"Generated {len(courses)} courses across {len(departments)} departments")

# ── 1b. Professors ──
# ~50 professors, each teaching 3-5 courses within their department.

professors = {}
W = {}

prof_id = 1
for dept, d_courses in dept_courses.items():
    remaining = list(d_courses)
    random.shuffle(remaining)
    while remaining:
        load = random.randint(3, 5)
        assigned = remaining[:load]
        remaining = remaining[load:]
        pname = f"Prof_{dept}_{prof_id:02d}"
        professors[pname] = assigned
        # weekly slot limit: slightly above their course count so it's
        # feasible but tight enough that workload penalties can trigger
        W[pname] = len(assigned) + random.randint(0, 2)
        prof_id += 1

print(f"Generated {len(professors)} professors")

# ── 1c. Rooms ──
# 10 rooms with a realistic capacity distribution

rooms = {
    "R01": 30,  # small seminar
    "R02": 30,  # small seminar
    "R03": 50,  # medium classroom
    "R04": 50,  # medium classroom
    "R05": 75,  # medium-large
    "R06": 75,  # medium-large
    "R07": 100,  # large classroom
    "R08": 120,  # large classroom
    "R09": 150,  # lecture hall
    "R10": 200,  # auditorium
}

# ── 1d. Academic programs ──
# 9 programs (3 per department).  Each program draws a core from its own
# department plus a few cross-listed courses from the other two depts.

programs = {}
for dept, d_courses in dept_courses.items():
    # split dept courses into 3 program tracks
    sorted_c = sorted(d_courses)
    third = len(sorted_c) // 3
    tracks = [
        sorted_c[:third],
        sorted_c[third:2 * third],
        sorted_c[2 * third:],
    ]
    track_names = ["Core", "Applied", "Advanced"]
    other_depts = [d for d in dept_courses if d != dept]
    for i, track in enumerate(tracks):
        # add 4-6 cross-listed courses from other departments
        cross = []
        for od in other_depts:
            cross += random.sample(
                dept_courses[od],
                min(random.randint(2, 3), len(dept_courses[od]))
            )
        prog_name = f"{dept}_{track_names[i]}"
        programs[prog_name] = track + cross
        print(f"  Program {prog_name}: {len(programs[prog_name])} courses")

# ── 1e. Timeslots ──
# 5 slots/day × 5 days = 25 weekly slots (each 2 hours)

num_days = 5
slots_per_day = 5
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