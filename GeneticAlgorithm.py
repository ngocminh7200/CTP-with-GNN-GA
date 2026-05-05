import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from DataIntegration import course_df
import random
import time
import matplotlib.pyplot as plt




#removing the duplicates because there are some changes to be done in preprocessing. Hence keeping the most simple version
#we have a key that is unique. it differentiates to the level of seminar group as well
#every key has a professor associated with it
#number of students for each key
#program of each key (our soft constraint works on this)

course_df = course_df.copy()
course_df = course_df.drop_duplicates(subset='Key', keep='first')
course_df = course_df.sort_values('Program').reset_index(drop=True)

schedule_df = course_df[['Key', 'Staff', 'NUMSTU', 'Program', 'Type', 'Module']].copy()
print(schedule_df.shape)

classrooms= pd.read_excel("Data/Teaching_rooms_2025_26.xlsx", header=2)
print(classrooms.shape)
#ROOMS
room_list = classrooms['Host Key'].tolist()
room_capacities = classrooms['Capacity'].tolist()

#room capacity lookup
room_capacity_lookup = dict(zip(range(len(room_list)), room_capacities))
print(room_capacity_lookup)   #148 rooms are given  unique numerical ID's

# Step 1: Define timeslots
days = ['M', 'Tu', 'W', 'Th', 'F']
slots = ['T1', 'T2', 'T3', 'T4', 'T5']
timeslots = [d + s for d in days for s in slots]
# ['MT1','MT2','MT3','MT4','MT5','TT1',...,'FT5'] — 25 total
print(f"Timeslots: {timeslots}")
print(f"Total: {len(timeslots)}")


course_list = course_df['Key'].tolist()
C = len(course_list)
print(f"Number of courses: {C}")



#To randomly assign classroom and time slots to a course

C = len(course_list)  # number of courses
T = 25                # timeslots
R = len(room_list)    # number of rooms

#given a course list (keys : prof, program, numstu)
def random_individual():
    timeslot_genes = np.random.randint(0, T, size=C)  # random timeslot for each course
    room_genes     = np.random.randint(0, R, size=C)  # random room for each course
    return np.stack([timeslot_genes, room_genes], axis=1)  # shape (C, 2)

#a sample gene is a course 
#an allel is the course allocation

#To convert a chromosome/schedule into a dataframe suitable for graph input

def decode_individual(ind):
    df = pd.DataFrame({
        'course index': np.arange(C),  #enumaerate the keys/course list
        'Key':          course_list,   
        'timeslot':     ind[:, 0] + 1,          # 1-indexed to match GNN
        'timeslot_label': [timeslots[i] for i in ind[:, 0]],  # readable label e.g. 'MT3'
        'classroom index': ind[:, 1],
        'room':         [room_list[i] for i in ind[:, 1]],    # readable label e.g. 'FAW.1.01'
        'capacity':     [room_capacities[i] for i in ind[:, 1]],
        'professor':    schedule_df['Staff'].tolist(),
        'numstu':       schedule_df['NUMSTU'].tolist(),
        'dl':           schedule_df['Program'].tolist(),
    })
    return df
    

# H1 - a course cannot have two timeslots (each course assigned exactly once)
def count_course_duplicates(schedule):
    course_counts = schedule.groupby('Key').size()
    duplicates = course_counts[course_counts > 1]
    H1 = int((course_counts > 1).sum())
    return H1, duplicates

# H2 - a professor cannot teach two courses at the same timeslot (professor conflict)
def count_professor_conflicts(schedule):
    prof_ts = schedule.groupby(['professor', 'timeslot']).size()
    conflicts = prof_ts[prof_ts > 1]
    H2 = int((prof_ts > 1).sum())
    return H2, conflicts

# H3 - a room cannot have two courses at the same timeslot (room conflict)
def count_room_conflicts(schedule):
    room_ts = schedule.groupby(['room', 'timeslot']).size()
    conflicts = room_ts[room_ts > 1]
    H3 = int((room_ts > 1).sum())
    return H3, conflicts

# H4 - a room capacity cannot be smaller than the number of students (capacity violation)
def count_capacity_violations(schedule):
    H4 = int((schedule['numstu'] > schedule['capacity']).sum())
    violations = schedule[schedule['numstu'] > schedule['capacity']]
    return H4, violations

# S1 - courses in the same program should not overlap at the same timeslot (program overlap)
def count_program_overlaps(schedule):
    prog_ts = schedule.groupby(['dl', 'timeslot']).size()
    conflicts = prog_ts[prog_ts > 1]
    S1 = int((prog_ts > 1).sum())
    return S1, conflicts

# S2 - total courses at a timeslot cannot exceed number of rooms (global overflow)
def count_global_overlap(schedule, num_rooms=R):
    courses_per_ts = schedule.groupby('timeslot').size()
    overflow = (courses_per_ts - num_rooms).clip(lower=0)
    S2 = int(overflow.sum())
    return S2, overflow[overflow > 0]



def get_course_penalties(schedule, num_rooms=R):
    course_penalty = np.zeros(len(schedule))
    # precompute groupby ONCE
    course_counts = schedule.groupby('Key').size()
    prof_ts = schedule.groupby(['professor', 'timeslot']).size()
    room_ts = schedule.groupby(['room', 'timeslot']).size()
    prog_ts = schedule.groupby(['dl', 'timeslot']).size()
    courses_per_ts = schedule.groupby('timeslot').size()

    for idx, row in schedule.iterrows():

        # H1 — hard — course duplicates
        key = row['Key']
        if course_counts[key] > 1:
            course_penalty[idx] += (course_counts[key] - 1) * 100

        # H2 — hard — professor conflict
        prof_key = (row['professor'], row['timeslot'])
        if prof_key in prof_ts.index and prof_ts[prof_key] > 1:
            course_penalty[idx] += (prof_ts[prof_key] - 1) * 100

        # H3 — hard — room conflict
        room_key = (row['room'], row['timeslot'])
        if room_key in room_ts.index and room_ts[room_key] > 1:
            course_penalty[idx] += (room_ts[room_key] - 1) * 100

        # H4 — hard — capacity overflow
        if row['numstu'] > row['capacity']:
            overflow = row['numstu'] - row['capacity']
            course_penalty[idx] += overflow * 100

        # S1 — soft — program overlap
        prog_key = (row['dl'], row['timeslot'])
        if prog_key in prog_ts.index and prog_ts[prog_key] > 1:
            course_penalty[idx] += (prog_ts[prog_key] - 1) * 0.1

        # S2 — soft — global overflow
        ts = row['timeslot']
        ov = courses_per_ts.get(ts, 0) - num_rooms
        if ov > 0:
            course_penalty[idx] += ov * 0.01

    return course_penalty

def fitness_function(schedule, num_rooms=R):

    # hard constraints
    H1, _ = count_course_duplicates(schedule)   # can never be violated
    H2, _ = count_professor_conflicts(schedule)
    H3, _ = count_room_conflicts(schedule)
    H4, _ = count_capacity_violations(schedule)

    # soft constraints
    S1, _ = count_program_overlaps(schedule)
    S2, _ = count_global_overlap(schedule, num_rooms)

    # weights
    w_H1 = 100   # hard
    w_H2 = 100   # hard
    w_H3 = 100   # hard
    w_H4 = 100   # hard
    w_S1 = 0.1   # soft — matches Gurobi beta
    w_S2 = 0.01  # soft — matches Gurobi gamma

    # scores
    soft_score   = (w_S1 * S1) + (w_S2 * S2)
    hard_penalty = (w_H1*H1) + (w_H2*H2) + (w_H3*H3) + (w_H4*H4)

    # fitness — minimize towards 0
    fitness = soft_score + hard_penalty

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


#as of now lets do a simple one. 
#hta paper  does is , it crossovers with a probabilities
# and wehen it does do the crossover, it does two different crossovers with two didfferent probabilities.

# find where each program starts and ends in course_df
program_boundaries = []
programs = course_df['Program'].tolist()

current_program = programs[0]
start_idx = 0

for i, prog in enumerate(programs):
    if prog != current_program:
        program_boundaries.append((current_program, start_idx, i))
        current_program = prog
        start_idx = i

# add last program
program_boundaries.append((current_program, start_idx, len(programs)))


#does th crossover
def program_crossover(parent1, parent2, program_boundaries=program_boundaries, num_points=3):
    c1 = np.zeros_like(parent1)
    c2 = np.zeros_like(parent2)
    
    # pick random program split points
    split_indices = sorted(random.sample(range(len(program_boundaries)), num_points))
    
    current_parent_c1 = parent1
    current_parent_c2 = parent2  # c2 starts from opposite parent
    prev_end = 0
    
    for idx in split_indices:
        prog, start, end = program_boundaries[idx]
        
        c1[prev_end:start] = current_parent_c1[prev_end:start]
        c2[prev_end:start] = current_parent_c2[prev_end:start]
        
        # switch parents
        current_parent_c1 = parent2 if current_parent_c1 is parent1 else parent1
        current_parent_c2 = parent1 if current_parent_c2 is parent2 else parent2
        
        prev_end = end
    
    c1[prev_end:] = current_parent_c1[prev_end:]
    c2[prev_end:] = current_parent_c2[prev_end:]
    
    return c1, c2


def cx_one_line(parent1, parent2):
    c1 = parent1.copy()
    c2 = parent2.copy()

    # pick random course index
    idx = random.randint(0, len(parent1) - 1)

    # swap that course's full assignment [timeslot, room]
    c1[idx] = parent2[idx]
    c2[idx] = parent1[idx]

    return c1, c2

#the paper does the crossover overall by using everyparent in pairs after randomly shuffling.
#amd the corssoverhappens with a probability

def crossover_all(population, program_boundaries=program_boundaries, probab_crossover=0.6):
    pop = list(population)
    random.shuffle(pop)

    offspring = []

    for i in range(0, len(pop), 2):
        p1 = pop[i]
        p2 = pop[i + 1]

        if random.random() < probab_crossover:
            if random.random() < 0.5:
                # program crossover → big change
                c1, c2 = program_crossover(p1, p2, program_boundaries, num_points=3)
            else:
                # cx_one_line → small change
                c1, c2 = cx_one_line(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()

        offspring.append(c1)
        offspring.append(c2)

    return offspring

N_MUTATIONS = int(0.01 * C)

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
        child[i, 0] = random.randint(0, T-1)
    return child

def mut_swap_rooms(child):
    child = child.copy()
    for _ in range(N_MUTATIONS // 2):
        i, j = random.sample(range(len(child)), 2)
        child[i, 1], child[j, 1] = child[j, 1], child[i, 1]
    return child

def mut_change_room(child):
    child = child.copy()
    indices = random.sample(range(len(child)), N_MUTATIONS)
    for i in indices:
        child[i, 1] = random.randint(0, R-1)
    return child
    
def mut_change_penalty(child):
    child = child.copy()
    child_df = decode_individual(child)
    
    # get per course penalty
    course_penalty = get_course_penalties(child_df)
    
    # find top N_MUTATIONS worst courses
    worst_indices = np.argsort(course_penalty)[-N_MUTATIONS:]
     
    for worst_idx in worst_indices:
        if random.random() < 0.6:
            child[worst_idx, 0] = random.randint(0, T-1)  # change timeslot
        else:
            child[worst_idx, 1] = random.randint(0, R-1)  # change room
    
    return child

#do the mutation as a  whole with probabilities
def mutation_all(offspring, probab_mutation=1.0):
    mutated = []
    for child in offspring:
        if random.random() < probab_mutation:
            r = random.random()
            if r < 0.2:
                child = mut_swap_timeslots(child)    # 20%
            elif r < 0.4:
                child = mut_change_timeslot(child)   # 20%
            elif r < 0.6:
                child = mut_swap_rooms(child)        # 20%
            elif r < 0.8:
                child = mut_change_room(child)       # 20%
            else:
                child = mut_change_penalty(child)    # 20%
        mutated.append(child)
    return mutated

#this mutated children are sent thoguh GNN for further mutation

# after the child is muatted through GNN, 
def crowding_distance(chrom1, chrom2):
    # number of positions where they differ
    # each position has 2 values [timeslot, room]
    timeslot_diff = (chrom1[:, 0] != chrom2[:, 0]).sum()
    room_diff = (chrom1[:, 1] != chrom2[:, 1]).sum()
    return timeslot_diff + room_diff

#does mapping for each parent to each child
def calc_crowding_distances(population, offspring):
    n = len(population)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = crowding_distance(population[i], offspring[j])
    
    return distances  # shape (n, n)



def find_matchings(distances):
    # linear sum assignment finds optimal one-to-one matching
    # minimizes total distance → most similar pairs matched
    row_indices, col_indices = linear_sum_assignment(distances)
    # matches[i] = which offspring matches parent i
    matches = list(zip(row_indices, col_indices))
    return matches

#now select between parent and the child
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
            p_child = parent_fitness / (parent_fitness + child_fitness)
            if random.random() < p_child:
                next_generation.append(child)
                next_scores.append(child_fitness)
            else:
                next_generation.append(parent)
                next_scores.append(parent_fitness)

    return next_generation, next_scores
#in the next gen after choosing , we would like to replace the worst oes with elite onesin the previous gen

def apply_elitism(next_generation, next_gen_scores, old_population, old_scores, n_elites=2):
    # get top n from OLD population
    elite_indices = sorted(range(len(old_scores)),
                           key=lambda x: old_scores[x])[:n_elites]
    elites = [old_population[i] for i in elite_indices]

    # find worst n in next generation
    worst_indices = sorted(range(len(next_generation)),
                           key=lambda x: next_gen_scores[x],
                           reverse=True)[:n_elites]

    # replace worst with elites
    for k, idx in enumerate(worst_indices):
        next_generation[idx] = elites[k]

    return next_generation

def update_prob_greedy(min_prob_greedy=0.4, epoch=0, max_epochs=100):
    prob_greedy = min_prob_greedy + (1.0 - min_prob_greedy) * (epoch / max_epochs)
    return min(prob_greedy, 1.0)


#minimum S1 contraint by pigeon hole principle
courses_per_program = {prog: (end - start) for prog, start, end in program_boundaries}
min_S1 = sum(max(0, count - 25) for count in courses_per_program.values())
print(f"Minimum possible S1: {min_S1}")




# To generate a population
POP_SIZE = 200
population = [random_individual() for _ in range(POP_SIZE)]
#import this very population in a dif.py file to main homegenity amongst Ga anf Ga+GNN

def print_best(best_chromosome, best_fitness, gen):
    best_df = decode_individual(best_chromosome)
    _, H1, H2, H3, H4, S1, S2 = fitness_function(best_df)
    print(f"\n Best chromosome (gen {gen}):")
    print(f"   fitness={best_fitness}, H1={H1}, H2={H2}, H3={H3}, H4={H4}, S1={S1}, S2={S2}")


def genetic_algorithm(population, n_gen=100, patience=30):

    best_fitness = float('inf')
    best_chromosome = None
    no_improve_count = 0
    prob_greedy = 0.4

    # tracking history
    history_fitness  = []
    history_hard     = []
    history_soft     = []
    history_gen_time = []

    total_start = time.time()

    for i in range(n_gen):


        gen_start = time.time()

        # evaluate population
        fitness_df, scores = evaluate_population(population)
        current_best = min(scores)
        current_best_idx = scores.index(current_best)
        best_row = fitness_df.iloc[current_best_idx]

        # track metrics
        hard = int(best_row['H1']) + int(best_row['H2']) + int(best_row['H3']) + int(best_row['H4'])
        soft = int(best_row['S1']) + int(best_row['S2'])
        history_fitness.append(current_best)
        history_hard.append(hard)
        history_soft.append(soft)

        # track best ever
        if current_best < best_fitness:
            best_fitness = current_best
            best_chromosome = population[current_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # add this before stopping conditions
        print(f"Gen {i + 1}: best={current_best}, "
              f"H1={int(best_row['H1'])}, "
              f"H2={int(best_row['H2'])}, "
              f"H3={int(best_row['H3'])}, "
              f"H4={int(best_row['H4'])}, "
              f"S1={int(best_row['S1'])}, "
              f"S2={int(best_row['S2'])}, "
              f"no_improve={no_improve_count}, "
              f"prob_greedy={prob_greedy:.2f}")


        # stopping condition 1: patience exceeded → return BEST EVER
        if no_improve_count >= patience:
            print(f" Patience exceeded at gen {i+1}")
            print_best(best_chromosome, best_fitness, i+1)
            history_gen_time.append(time.time() - gen_start)
            break

        # stopping condition 2: max generations → return BEST EVER
        if i == n_gen - 1:
            print(f" Max generations reached")
            print_best(best_chromosome, best_fitness, i+1)
            history_gen_time.append(time.time() - gen_start)
            break

        # ── stopping condition 3: hard=0 and S1=min → optimal ──
        if (int(best_row['H1'])==0 and
            int(best_row['H2'])==0 and
            int(best_row['H3'])==0 and
            int(best_row['H4'])==0 and
            int(best_row['S1']) <= min_S1):
            print(f" Optimal solution found at gen {i+1}!")
            print_best(population[current_best_idx], current_best, i+1)
            history_gen_time.append(time.time() - gen_start)
            break

        # step 1: crossover
        offspring = crossover_all(population, program_boundaries, probab_crossover=0.6)

        # step 2: mutation
        offspring = mutation_all(offspring)

        # ── step 3: GNN (when ready) ──
        # offspring = GNN_improvement(offspring)

        # step 4: evaluate offspring
        offspring_fitness_df, offspring_scores = evaluate_population(offspring)

        #  step 5: crowding distances + matching
        distances = calc_crowding_distances(population, offspring)
        matches = find_matchings(distances)

        # step 6: selection
        old_population = list(population)
        old_scores = list(scores)

        population, selected_scores = selection(population, offspring, matches,
                                                scores, offspring_scores, prob_greedy)

        population = apply_elitism(population, selected_scores,
                                   old_population, old_scores, n_elites=2)

        # step 8: update prob_greedy
        prob_greedy = update_prob_greedy(min_prob_greedy=0.4, epoch=i, max_epochs=n_gen)

        # track generation time
        history_gen_time.append(time.time() - gen_start)

    # total time
    total_time = time.time() - total_start
    print(f"\n Total time: {total_time:.2f}s ({total_time/60:.2f} mins)")

    # plots
    gens = range(1, len(history_fitness) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GA Performance', fontsize=16)

    # fitness
    axes[0, 0].plot(gens, history_fitness, color='blue')
    axes[0, 0].set_title('Best Fitness per Generation')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].grid(True)

    # hard constraints
    axes[0, 1].plot(gens, history_hard, color='red')
    axes[0, 1].set_title('Hard Constraint Violations')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Total Hard Violations')
    axes[0, 1].grid(True)

    # soft constraints
    axes[1, 0].plot(gens, history_soft, color='green')
    axes[1, 0].set_title('Soft Constraint Violations')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Total Soft Violations')
    axes[1, 0].grid(True)

    # generation time
    axes[1, 1].plot(gens, history_gen_time, color='orange')
    axes[1, 1].set_title('Time per Generation')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    return best_chromosome, history_fitness, history_hard, history_soft, history_gen_time

best_chromosome, history_fitness, history_hard, history_soft, history_gen_time = genetic_algorithm(population, n_gen=100, patience=20)
best_schedule = decode_individual(best_chromosome)
print(best_schedule)
