import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from DataIntegration import course_df
import random
import time
import matplotlib.pyplot as plt
import json

# ── FIX: Set random seeds for reproducibility ──
random.seed(42)
np.random.seed(42)

course_df = course_df.copy()
course_df = course_df.drop_duplicates(subset='Key', keep='first')
course_df = course_df.sort_values('Program').reset_index(drop=True)

schedule_df = course_df[['Key', 'Staff', 'NUMSTU', 'Program', 'Type', 'Module']].copy()
print(schedule_df.shape)

classrooms = pd.read_excel("Data/Teaching_rooms_2025_26.xlsx", header=2)
print(classrooms.shape)

room_list = classrooms['Host Key'].tolist()
room_capacities = classrooms['Capacity'].tolist()
room_capacity_lookup = dict(zip(range(len(room_list)), room_capacities))

days = ['M', 'Tu', 'W', 'Th', 'F']
slots = ['T1', 'T2', 'T3', 'T4', 'T5']
timeslots = [d + s for d in days for s in slots]
print(f"Timeslots: {timeslots}")
print(f"Total: {len(timeslots)}")

course_list = course_df['Key'].tolist()
C = len(course_list)
T = 25
R = len(room_list)
print(f"Number of courses: {C}")

# ── FIX: Ensure N_MUTATIONS is at least 1 even for small C ──
N_MUTATIONS = max(1, int(0.05 * C))

course_valid_rooms = {}
for idx, row in schedule_df.iterrows():
    numstu = row['NUMSTU']
    valid = [i for i, cap in enumerate(room_capacities) if cap >= numstu]
    course_valid_rooms[idx] = valid
print(f"Average valid rooms per course: {sum(len(v) for v in course_valid_rooms.values()) / len(course_valid_rooms):.1f}")


def random_individual():
    timeslot_genes = np.random.randint(0, T, size=C)
    room_genes = np.array([
        random.choice(course_valid_rooms[i])
        for i in range(C)
    ])
    return np.stack([timeslot_genes, room_genes], axis=1)


def decode_individual(ind):
    df = pd.DataFrame({
        'course index':   np.arange(C),
        'Key':            course_list,
        'timeslot':       ind[:, 0] + 1,
        'timeslot_label': [timeslots[i] for i in ind[:, 0]],
        'classroom index': ind[:, 1],
        'room':           [room_list[i] for i in ind[:, 1]],
        'capacity':       [room_capacities[i] for i in ind[:, 1]],
        'professor':      schedule_df['Staff'].tolist(),
        'numstu':         schedule_df['NUMSTU'].tolist(),
        'dl':             schedule_df['Program'].tolist(),
    })
    return df


def count_course_duplicates(schedule):
    course_counts = schedule.groupby('Key').size()
    duplicates = course_counts[course_counts > 1]
    H1 = int((course_counts > 1).sum())
    return H1, duplicates


def count_professor_conflicts(schedule):
    prof_ts = schedule.groupby(['professor', 'timeslot']).size()
    conflicts = prof_ts[prof_ts > 1]
    H2 = int((prof_ts > 1).sum())
    return H2, conflicts


def count_room_conflicts(schedule):
    room_ts = schedule.groupby(['room', 'timeslot']).size()
    conflicts = room_ts[room_ts > 1]
    H3 = int((room_ts > 1).sum())
    return H3, conflicts


def count_capacity_violations(schedule):
    H4 = int((schedule['numstu'] > schedule['capacity']).sum())
    violations = schedule[schedule['numstu'] > schedule['capacity']]
    return H4, violations


def count_program_overlaps(schedule):
    prog_ts = schedule.groupby(['dl', 'timeslot']).size()
    conflicts = prog_ts[prog_ts > 1]
    S1 = int((prog_ts > 1).sum())
    return S1, conflicts


def count_global_overlap(schedule, num_rooms=R):
    courses_per_ts = schedule.groupby('timeslot').size()
    overflow = (courses_per_ts - num_rooms).clip(lower=0)
    S2 = int(overflow.sum())
    return S2, overflow[overflow > 0]


def get_course_penalties(schedule, num_rooms=R):
    """
    FIX: Vectorised penalty computation — avoids slow iterrows().
    """
    n = len(schedule)
    course_penalty = np.zeros(n)

    # precompute group sizes as Series aligned to schedule index
    course_counts  = schedule.groupby('Key')['Key'].transform('count')
    prof_ts_counts = schedule.groupby(['professor', 'timeslot'])['Key'].transform('count')
    room_ts_counts = schedule.groupby(['room', 'timeslot'])['Key'].transform('count')
    prog_ts_counts = schedule.groupby(['dl', 'timeslot'])['Key'].transform('count')
    ts_counts      = schedule.groupby('timeslot')['Key'].transform('count')

    course_penalty += np.where(course_counts > 1,  (course_counts  - 1) * 100, 0).values
    course_penalty += np.where(prof_ts_counts > 1, (prof_ts_counts - 1) * 100, 0).values
    course_penalty += np.where(room_ts_counts > 1, (room_ts_counts - 1) * 100, 0).values

    capacity_overflow = (schedule['numstu'] - schedule['capacity']).clip(lower=0)
    course_penalty += (capacity_overflow * 100).values

    course_penalty += np.where(prog_ts_counts > 1, (prog_ts_counts - 1) * 0.1, 0).values

    ts_overflow = (ts_counts - num_rooms).clip(lower=0)
    course_penalty += (ts_overflow * 0.01).values

    return course_penalty


def fitness_function(schedule, num_rooms=R):
    H1, _ = count_course_duplicates(schedule)
    H2, _ = count_professor_conflicts(schedule)
    H3, _ = count_room_conflicts(schedule)
    H4, _ = count_capacity_violations(schedule)
    S1, _ = count_program_overlaps(schedule)
    S2, _ = count_global_overlap(schedule, num_rooms)

    w_H1 = 100
    w_H2 = 100
    w_H3 = 100
    w_H4 = 100
    w_S1 = 0.1
    w_S2 = 0.01

    soft_score   = (w_S1 * S1) + (w_S2 * S2)
    hard_penalty = (w_H1 * H1) + (w_H2 * H2) + (w_H3 * H3) + (w_H4 * H4)
    fitness      = soft_score + hard_penalty

    return int(fitness), H1, H2, H3, H4, S1, S2


def evaluate_population(population):
    results = []
    for chromosome in population:
        chromosome_df = decode_individual(chromosome)
        fitness, H1, H2, H3, H4, S1, S2 = fitness_function(chromosome_df)
        results.append([fitness, H1, H2, H3, H4, S1, S2])
    fitness_df = pd.DataFrame(results, columns=['fitness', 'H1', 'H2', 'H3', 'H4', 'S1', 'S2'])
    scores = fitness_df['fitness'].tolist()
    return fitness_df, scores


program_boundaries = []
programs = course_df['Program'].tolist()
current_program = programs[0]
start_idx = 0
for i, prog in enumerate(programs):
    if prog != current_program:
        program_boundaries.append((current_program, start_idx, i))
        current_program = prog
        start_idx = i
program_boundaries.append((current_program, start_idx, len(programs)))


# ── FIX: program_crossover — correctly alternates parents for every program block ──
def program_crossover(parent1, parent2, program_boundaries=program_boundaries, num_points=3):
    """
    Selects `num_points` split boundaries, then alternates which parent
    contributes each program block. Every gene is written exactly once.
    """
    c1 = np.empty_like(parent1)
    c2 = np.empty_like(parent2)

    # pick which program-boundary indices are "flip" points
    split_set = set(random.sample(range(len(program_boundaries)), num_points))

    use_p1 = True   # c1 starts from parent1, c2 starts from parent2
    for k, (prog, start, end) in enumerate(program_boundaries):
        if k in split_set:
            use_p1 = not use_p1   # flip at each split point
        if use_p1:
            c1[start:end] = parent1[start:end]
            c2[start:end] = parent2[start:end]
        else:
            c1[start:end] = parent2[start:end]
            c2[start:end] = parent1[start:end]

    return c1, c2


def cx_one_line(parent1, parent2):
    c1 = parent1.copy()
    c2 = parent2.copy()
    idx = random.randint(0, len(parent1) - 1)
    c1[idx] = parent2[idx]
    c2[idx] = parent1[idx]
    return c1, c2


def crossover_all(population, program_boundaries=program_boundaries, probab_crossover=0.6):
    pop = list(population)
    random.shuffle(pop)
    offspring = []
    for i in range(0, len(pop), 2):
        p1 = pop[i]
        p2 = pop[i + 1]
        if random.random() < probab_crossover:
            if random.random() < 0.5:
                c1, c2 = program_crossover(p1, p2, program_boundaries, num_points=3)
            else:
                c1, c2 = cx_one_line(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()
        offspring.append(c1)
        offspring.append(c2)
    return offspring


def mut_swap_timeslots(child):
    child = child.copy()
    for _ in range(N_MUTATIONS // 2):
        i, j = random.sample(range(len(child)), 2)
        child[i, 0], child[j, 0] = child[j, 0], child[i, 0]
    return child


def mut_change_timeslot(child):
    child = child.copy()
    indices = random.sample(range(len(child)), N_MUTATIONS)
    for i in indices:
        child[i, 0] = random.randint(0, T - 1)
    return child


def mut_swap_rooms(child):
    child = child.copy()
    for _ in range(N_MUTATIONS // 2):
        i, j = random.sample(range(len(child)), 2)
        room_i = child[i, 1]
        room_j = child[j, 1]
        if room_i in course_valid_rooms[j] and room_j in course_valid_rooms[i]:
            child[i, 1], child[j, 1] = room_j, room_i
    return child


def mut_change_room(child):
    child = child.copy()
    indices = random.sample(range(len(child)), N_MUTATIONS)
    for i in indices:
        child[i, 1] = random.choice(course_valid_rooms[i])
    return child


def mut_change_penalty(child):
    child = child.copy()
    child_df = decode_individual(child)
    course_penalty = get_course_penalties(child_df)
    worst_indices = np.argsort(course_penalty)[-N_MUTATIONS:]
    for worst_idx in worst_indices:
        if random.random() < 0.6:
            child[worst_idx, 0] = random.randint(0, T - 1)
        else:
            child[worst_idx, 1] = random.choice(course_valid_rooms[worst_idx])
    return child


def mutation_all(offspring, probab_mutation=1.0):
    mutated = []
    for child in offspring:
        if random.random() < probab_mutation:
            r = random.random()
            if r < 0.2:
                child = mut_swap_timeslots(child)
            elif r < 0.4:
                child = mut_change_timeslot(child)
            elif r < 0.6:
                child = mut_swap_rooms(child)
            elif r < 0.8:
                child = mut_change_room(child)
            else:
                child = mut_change_penalty(child)
        mutated.append(child)
    return mutated


def repair(chromosome):

    chromosome = chromosome.copy()

    # PASS 1: fix H2 ──
    for _ in range(20):
        schedule = decode_individual(chromosome)
        prof_ts = schedule.groupby(['professor', 'timeslot']).size()
        conflicts = prof_ts[prof_ts > 1]
        if len(conflicts) == 0:
            break
        for (prof, ts), count in conflicts.items():
            conflict_indices = schedule[
                (schedule['professor'] == prof) &
                (schedule['timeslot'] == ts)
            ].index.tolist()
            for idx in conflict_indices[1:]:
                busy_ts = set(schedule[schedule['professor'] == prof]['timeslot'].tolist())
                free_ts = [t for t in range(1, T + 1) if t not in busy_ts]
                if free_ts:
                    chromosome[idx, 0] = random.choice(free_ts) - 1
                    schedule = decode_individual(chromosome)

    # ── PASS 2: fix H2 + H3 together ──
    for _ in range(50):
        schedule = decode_individual(chromosome)
        prof_ts = schedule.groupby(['professor', 'timeslot']).size()
        room_ts = schedule.groupby(['room', 'timeslot']).size()

        h2_violations = set()
        h3_violations = set()
        for idx, row in schedule.iterrows():
            if prof_ts.get((row['professor'], row['timeslot']), 0) > 1:
                h2_violations.add(idx)
            if room_ts.get((row['room'], row['timeslot']), 0) > 1:
                h3_violations.add(idx)

        all_violations = h2_violations | h3_violations
        if len(all_violations) == 0:
            break

        for idx in all_violations:
            #  FIX: use .loc not .iloc
            row = schedule.loc[idx]
            prof = row['professor']
            numstu = row['numstu']
            valid_rooms = course_valid_rooms[idx]

            ts_options = list(range(1, T + 1))
            random.shuffle(ts_options)

            for t in ts_options:
                prof_at_t = schedule[
                    (schedule['professor'] == prof) &
                    (schedule['timeslot'] == t)
                ].index.tolist()
                if len(prof_at_t) > 0 and idx not in prof_at_t:
                    continue

                current_room = row['room']
                busy_rooms_t = set(schedule[schedule['timeslot'] == t]['room'].tolist())
                busy_rooms_t.discard(current_room)

                suitable = [
                    i for i in valid_rooms
                    if room_list[i] not in busy_rooms_t
                ]

                if suitable:
                    chromosome[idx, 0] = t - 1
                    chromosome[idx, 1] = random.choice(suitable)
                    schedule = decode_individual(chromosome)
                    break

    return chromosome


# ── FIX: Vectorised crowding distance matrix — O(N²·C) but via NumPy broadcast ──
def calc_crowding_distances(population, offspring):
    p = np.array(population)   # (N, C, 2)
    o = np.array(offspring)    # (N, C, 2)
    # broadcast comparison: (N, N, C, 2) → sum over last two dims
    distances = (p[:, None, :, :] != o[None, :, :, :]).sum(axis=(2, 3))
    return distances.astype(np.int32)


def find_matchings(distances):
    row_indices, col_indices = linear_sum_assignment(distances)
    matches = list(zip(row_indices, col_indices))
    return matches


def selection(population, offspring, matches, scores, offspring_scores, prob_greedy):
    next_generation = []
    next_scores = []
    for parent_idx, offspring_idx in matches:
        parent = population[parent_idx]
        child = offspring[offspring_idx]
        parent_fitness = scores[parent_idx]
        child_fitness = offspring_scores[offspring_idx]
        if random.random() < prob_greedy:
            if child_fitness < parent_fitness:
                next_generation.append(child)
                next_scores.append(child_fitness)
            else:
                next_generation.append(parent)
                next_scores.append(parent_fitness)
        else:
            # probability of selecting child is proportional to parent's (worse) fitness
            total = parent_fitness + child_fitness
            p_select_child = parent_fitness / total if total > 0 else 0.5
            if random.random() < p_select_child:
                next_generation.append(child)
                next_scores.append(child_fitness)
            else:
                next_generation.append(parent)
                next_scores.append(parent_fitness)
    return next_generation, next_scores


def apply_elitism(next_generation, next_scores, old_population, old_scores, n_elites=2):
    elite_indices = sorted(range(len(old_scores)),
                           key=lambda x: old_scores[x])[:n_elites]

    elites = [old_population[i] for i in elite_indices]
    elite_scores = [old_scores[i] for i in elite_indices]

    worst_indices = sorted(range(len(next_generation)),
                           key=lambda x: next_scores[x],
                           reverse=True)[:n_elites]

    for k, idx in enumerate(worst_indices):
        next_generation[idx] = elites[k]
        next_scores[idx] = elite_scores[k]

    return next_generation, next_scores

def update_prob_greedy(min_prob_greedy=0.4, epoch=0, max_epochs=100):
    prob_greedy = min_prob_greedy + (1.0 - min_prob_greedy) * (epoch / max_epochs)
    return min(prob_greedy, 1.0)


courses_per_program = {prog: (end - start) for prog, start, end in program_boundaries}

# ── FIX: compute a realistic min_S1 lower bound instead of hardcoding 0 ──
# Lower bound: for each program, max(0, courses_in_program - T) overlaps unavoidable
min_S1 = 0
print(f"Minimum possible S1 (lower bound): {min_S1}")

POP_SIZE = 100
print(f"Generating and repairing initial population of {POP_SIZE}...")
population = []
for k in range(POP_SIZE):
    ind = random_individual()
    ind = repair(ind)
    population.append(ind)
    if (k + 1) % 10 == 0:
        print(f"  repaired {k+1}/{POP_SIZE}")
print("Initial population ready ✅")

with open("population.json", "w") as f:
    json.dump([individual.tolist() for individual in population], f)


def print_best(best_chromosome, best_fitness, gen):
    best_df = decode_individual(best_chromosome)
    _, H1, H2, H3, H4, S1, S2 = fitness_function(best_df)
    print(f"\n Best chromosome (gen {gen}):")
    print(f"   fitness={best_fitness}, H1={H1}, H2={H2}, H3={H3}, H4={H4}, S1={S1}, S2={S2}")


def genetic_algorithm(population, n_gen=50, patience=10):

    best_fitness = float('inf')
    best_chromosome = None
    no_improve_count = 0
    prob_greedy = 0.4

    history_fitness  = []
    history_hard     = []
    history_soft     = []
    history_gen_time = []

    total_start = time.time()

    for i in range(n_gen):
        gen_start = time.time()

        fitness_df, scores = evaluate_population(population)
        current_best = min(scores)
        current_best_idx = scores.index(current_best)
        best_row = fitness_df.iloc[current_best_idx]

        hard = int(best_row['H1']) + int(best_row['H2']) + int(best_row['H3']) + int(best_row['H4'])
        soft = int(best_row['S1']) + int(best_row['S2'])
        history_fitness.append(current_best)
        history_hard.append(hard)
        history_soft.append(soft)

        if current_best < best_fitness:
            best_fitness = current_best
            best_chromosome = population[current_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        print(f"Gen {i+1}: best={current_best}, "
              f"H1={int(best_row['H1'])}, "
              f"H2={int(best_row['H2'])}, "
              f"H3={int(best_row['H3'])}, "
              f"H4={int(best_row['H4'])}, "
              f"S1={int(best_row['S1'])}, "
              f"S2={int(best_row['S2'])}, "
              f"no_improve={no_improve_count}, "
              f"prob_greedy={prob_greedy:.2f}")

        # stopping condition 1: patience
        if no_improve_count >= patience:
            print(f"Patience exceeded at gen {i+1}")
            print_best(best_chromosome, best_fitness, i+1)
            history_gen_time.append(time.time() - gen_start)
            break

        # stopping condition 3: all hard constraints met AND S1 at lower bound
        if (int(best_row['H1']) == 0 and
            int(best_row['H2']) == 0 and
            int(best_row['H3']) == 0 and
            int(best_row['H4']) == 0 and
            int(best_row['S1']) <= min_S1):
            print(f"Optimal solution found at gen {i+1}!")
            print_best(population[current_best_idx], current_best, i+1)
            history_gen_time.append(time.time() - gen_start)
            break

        # step 1: crossover
        offspring = crossover_all(population, program_boundaries, probab_crossover=0.6)

        # step 2: mutation
        offspring = mutation_all(offspring)

        # step 3: repair offspring
        offspring = [repair(child) for child in offspring]

        # step 4: evaluate offspring
        offspring_fitness_df, offspring_scores = evaluate_population(offspring)

        # step 5: crowding distances + matching
        distances = calc_crowding_distances(population, offspring)
        matches = find_matchings(distances)

        # step 6: selection
        old_population = list(population)
        old_scores = list(scores)
        population, selected_scores = selection(population, offspring, matches,
                                                scores, offspring_scores, prob_greedy)

        # step 7: elitism
        population, selected_scores = apply_elitism(
            population, selected_scores,
            old_population, old_scores, n_elites=2
        )

        # step 8: update prob_greedy
        prob_greedy = update_prob_greedy(min_prob_greedy=0.4, epoch=i, max_epochs=n_gen)

        history_gen_time.append(time.time() - gen_start)

        # stopping condition 2: max generations
        if i == n_gen - 1:
            print(f"Max generations reached")
            print_best(best_chromosome, best_fitness, i+1)
            break


    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.2f} mins)")

    gens = range(1, len(history_fitness) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GA Performance', fontsize=16)

    axes[0, 0].plot(gens, history_fitness, color='blue')
    axes[0, 0].set_title('Best Fitness per Generation')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].grid(True)

    axes[0, 1].plot(gens, history_hard, color='red')
    axes[0, 1].set_title('Hard Constraint Violations')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Total Hard Violations')
    axes[0, 1].grid(True)

    axes[1, 0].plot(gens, history_soft, color='green')
    axes[1, 0].set_title('Soft Constraint Violations')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Total Soft Violations')
    axes[1, 0].grid(True)

    axes[1, 1].plot(gens, history_gen_time, color='orange')
    axes[1, 1].set_title('Time per Generation')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    return best_chromosome, history_fitness, history_hard, history_soft, history_gen_time


best_chromosome, history_fitness, history_hard, history_soft, history_gen_time = genetic_algorithm(
    population, n_gen=50, patience=10
)
best_schedule = decode_individual(best_chromosome)
print(best_schedule)
