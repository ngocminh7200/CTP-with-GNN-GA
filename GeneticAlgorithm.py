import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment



course_df = pd.read_excel(r"/Users/t.reddy/Library/CloudStorage/OneDrive-LondonSchoolofEconomics/Desktop/course_df.xlsx")
#removing the duplicates because there are some changes to be done in preprocessing. Hence keeping the most simple version
#we have a key that is unique. it differentiates to the level of seminar group as well
#every key has a professor associated with it
#number of students for each key
#program of each key (our soft constraint works on this)


course_df = course_df.drop_duplicates(subset='Key', keep='first')

schedule_df = course_df[['Key', 'Staff', 'NUMSTU', 'Program', 'Type', 'Module']].copy()


print(schedule_df.shape)
classrooms= pd.read_excel(r"/Users/t.reddy/Downloads/Teaching_rooms_2025_26.xlsx", header=2)
print(classrooms.shape)


#ROOMS
room_list = classrooms['Host Key'].tolist()
room_capacities = classrooms['Capacity'].tolist()

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
    




#constraints
def count_course_duplicates(schedule):
    # each course should appear exactly once
    course_counts = schedule.groupby('Key').size()
    
    duplicates = course_counts[course_counts > 1]
    H1 = (course_counts > 1).sum()
    
    return H1, duplicates
def count_professor_conflicts(schedule):
    # group by professor AND timeslot, count courses in each group
    prof_ts = schedule.groupby(['professor', 'timeslot']).size()
    
    # any group with more than 1 course = conflict
    conflicts = prof_ts[prof_ts > 1]
    
    # total number of conflicts
    H2 = (prof_ts > 1).sum()
    
    return H2, conflicts

def count_room_conflicts(schedule):
    # group by room AND timeslot, count courses in each group
    room_ts = schedule.groupby(['room', 'timeslot']).size()
    
    # any group with more than 1 course = conflict
    conflicts = room_ts[room_ts > 1]
    
    # total number of conflicts
    H3 = (room_ts > 1).sum()
    
    return H3, conflicts
def count_capacity_violations(schedule):
    # directly compare numstu vs capacity for each course
    violations = schedule[schedule['numstu'] > schedule['capacity']]
    
    # total number of violations
    H4 = (schedule['numstu'] > schedule['capacity']).sum()
    
    return H4, violations

def count_program_overlaps(schedule):
    # group by program AND timeslot, count courses in each group
    prog_ts = schedule.groupby(['dl', 'timeslot']).size()
    
    # any group with more than 1 course = overlap
    conflicts = prog_ts[prog_ts > 1]
    
    # total number of overlaps
    S2 = (prog_ts > 1).sum()
    
    return S2, conflicts


def count_global_overlap(schedule, num_rooms=R):
    # count total courses per timeslot
    courses_per_ts = schedule.groupby('timeslot').size()
    
    # overflow = courses beyond available rooms
    overflow = (courses_per_ts - num_rooms).clip(lower=0)
    
    S3 = int(overflow.sum())
    
    return S3, overflow[overflow > 0]


def get_course_penalties(schedule, num_rooms=R):
    course_penalty = np.zeros(len(schedule))
    
    # precompute groupby ONCE
    course_counts  = schedule.groupby('Key').size()
    prof_ts        = schedule.groupby(['professor', 'timeslot']).size()
    room_ts        = schedule.groupby(['classroom index', 'timeslot']).size()
    prog_ts        = schedule.groupby(['dl', 'timeslot']).size()
    courses_per_ts = schedule.groupby('timeslot').size()
    
    for idx, row in schedule.iterrows():
        
        # H1 — hard — course duplicates
        key = row['Key']
        if course_counts[key] > 1:
            course_penalty[idx] += (course_counts[key] - 1) * 10
        
        # H2 — hard — professor conflict
        prof_key = (row['professor'], row['timeslot'])
        if prof_key in prof_ts.index and prof_ts[prof_key] > 1:
            course_penalty[idx] += (prof_ts[prof_key] - 1) * 10
        
        # H3 — hard — room conflict
        room_key = (row['classroom index'], row['timeslot'])
        if room_key in room_ts.index and room_ts[room_key] > 1:
            course_penalty[idx] += (room_ts[room_key] - 1) * 10
        
        # H4 — hard — capacity overflow
        if row['numstu'] > row['capacity']:
            overflow = row['numstu'] - row['capacity']
            course_penalty[idx] += overflow * 10
        
        # S2 — soft — program overlap
        prog_key = (row['dl'], row['timeslot'])
        if prog_key in prog_ts.index and prog_ts[prog_key] > 1:
            course_penalty[idx] += (prog_ts[prog_key] - 1) * 1
        
        # S3 — soft — global overflow
        ts = row['timeslot']
        if ts in courses_per_ts.index:
            ov = courses_per_ts[ts] - num_rooms
            if ov > 0:
                course_penalty[idx] += ov * 1
    
    return course_penalty


def fitness_function(schedule, num_rooms=R):
    # hard constraints (violations)
    H1, _ = count_course_duplicates(schedule)  #can never be violated because of the way we design the dataframe.
    H2, _ = count_professor_conflicts(schedule)
    H3, _ = count_room_conflicts(schedule)
    H4, _ = count_capacity_violations(schedule)
    
    # soft constraints (violations)
    S2, _ = count_program_overlaps(schedule)
    S3, _ = count_global_overlap(schedule, num_rooms)
    
    # weights
    w_H1 = 10    # course duplicate
    w_H2 = 10    # professor conflict
    w_H3 = 10    # room conflict
    w_H4 = 10    # capacity violation
    w_S2 = 0.5     # program overlap
    w_S3 = 0.5   # global timeslot overflow
    
    # soft constraint score
    soft_score = (w_S2 * S2) + (w_S3 * S3)
    
    # hard constraint penalty (squared → always hurts a lot)
    hard_penalty = (w_H1 * H1) + (w_H2 * H2) + (w_H3 * H3) + (w_H4 * H4)
    
    # fitness = make as close to 0 as possible
    fitness = soft_score + (hard_penalty ** 2)
    
    return int(fitness), int(H1), int(H2), int(H3), int(H4), int(S2), int(S3)

#as of now lets do a simple one. 
#hta paper  does is , it crossovers with a popribabilit.
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
def program_crossover(parent1, parent2, program_boundaries, num_points=3):
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
        
        prev_end = start
    
    c1[prev_end:] = current_parent_c1[prev_end:]
    c2[prev_end:] = current_parent_c2[prev_end:]
    
    return c1, c2

#the paper does the crossover overall by using everyparent in pairs after randomly shuffling.
#amd the corssoverhappens with a probability

def crossover_all(population, program_boundaries, probab_crossover=0.6):
    # shuffle so pairing is random each generation
    pop = list(population)
    random.shuffle(pop)
    
    offspring = []
    
    # pop_size/2 pairs → each produces 2 children
    for i in range(0, len(pop), 2):
        p1 = pop[i]
        p2 = pop[i+1]
        
        if random.random() < probab_crossover:
            # crossover → 2 new children
            c1, c2 = program_crossover(p1, p2, program_boundaries, num_points=3)
        else:
            # no crossover → copies of parents
            c1, c2 = p1.copy(), p2.copy()
        
        offspring.append(c1)
        offspring.append(c2)
    
    return offspring  # same size as population



N_MUTATIONS = int(0.05 * C)  # ~139 courses

def mut_swap_timeslots(child):
    child = child.copy()
    for i in range(N_MUTATIONS // 2):
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
    for i in range(N_MUTATIONS // 2):
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
    
    for parent_idx, offspring_idx in matches:
        parent = population[parent_idx]
        child = offspring[offspring_idx]
        
        parent_fitness = scores[parent_idx]
        child_fitness = offspring_scores[offspring_idx]
        
        if random.random() < prob_greedy:
            # GREEDY → always pick better one
            if child_fitness < parent_fitness:
                next_generation.append(child)
            else:
                next_generation.append(parent)
        else:
            # PROBABILISTIC → worse can sometimes win
            p_child = parent_fitness / (parent_fitness + child_fitness)
            if random.random() < p_child:
                next_generation.append(child)
            else:
                next_generation.append(parent)
    
    return next_generation

#in the next gen after choosing , we would like to replace the worst oes with elite onesin the previous gen
def apply_elitism(next_generation, population, scores, n_elites=2):
    # get top n from current population
    elite_indices = sorted(range(len(scores)), 
                          key=lambda x: scores[x])[:n_elites]
    elites = [population[i] for i in elite_indices]
    
    # replace n worst in next generation with elites
    next_gen_scores = [fitness_function(decode_individual(c))[0] 
                      for c in next_generation]
    worst_indices = sorted(range(len(next_generation)),
                          key=lambda x: next_gen_scores[x],
                          reverse=True)[:n_elites]
    
    for k, idx in enumerate(worst_indices):
        next_generation[idx] = elites[k]
    
    return next_generation

def update_prob_greedy(min_prob_greedy=0.4, epoch=0, max_epochs=100):
    prob_greedy = min_prob_greedy + (1.0 - min_prob_greedy) * (epoch / max_epochs)
    return min(prob_greedy, 1.0)

















# To generate a population
POP_SIZE = 200
population = np.array([random_individual() for i in range(POP_SIZE)])


def genetic_algorithm(population, n_gen=100, patience=20):
    
    best_fitness = float('inf')
    best_chromosome = None
    no_improve_count = 0
    history = []
    prob_greedy = 0.4

    for i in range(n_gen):

        # evaluate population
        fitness_df, scores = evaluate_population(population)
        current_best = min(scores)
        current_best_idx = scores.index(current_best)
        history.append(current_best)

        # track best ever
        if current_best < best_fitness:
            best_fitness = current_best
            best_chromosome = population[current_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # print progress
        best_row = fitness_df.iloc[current_best_idx]
        print(f"Gen {i+1}: best={current_best}, "
              f"H1={int(best_row['H1'])}, "
              f"H2={int(best_row['H2'])}, "
              f"H3={int(best_row['H3'])}, "
              f"H4={int(best_row['H4'])}, "
              f"S2={int(best_row['S2'])}, "
              f"S3={int(best_row['S3'])}, "
              f"no_improve={no_improve_count}, "
              f"prob_greedy={prob_greedy:.2f}")

        # condition 3 — hard constraints = 0 → return CURRENT
        if (int(best_row['H1'])==0 and 
            int(best_row['H2'])==0 and 
            int(best_row['H3'])==0 and 
            int(best_row['H4'])==0):
            print(f"All hard constraints satisfied at gen {i+1}!")
            return population[current_best_idx], fitness_df, history

        # condition 1 — patience exceeded → return BEST EVER
        if no_improve_count >= patience:
            print(f" Patience exceeded at gen {i+1} — returning best ever found")
            return best_chromosome, fitness_df, history

        # condition 2 — max generations → return BEST EVER
        if i == n_gen - 1:
            print(f" Max generations reached — returning best ever found")
            return best_chromosome, fitness_df, history

        # step 1: crossover
        offspring = crossover_all(population, program_boundaries, probab_crossover=0.6)

        # step 2: mutation
        offspring = mutation_all(offspring)

        # step 3: GNN (when ready)
        # offspring = GNN_improvement(offspring)

        # step 4: evaluate offspring
        offspring_fitness_df, offspring_scores = evaluate_population(offspring)

        # step 5: crowding distances + matching
        distances = calc_crowding_distances(population, offspring)
        matches = find_matchings(distances)

        # step 6: selection
        population = selection(population, offspring, matches,
                              scores, offspring_scores, prob_greedy)

        # step 7: elitism
        population = apply_elitism(population, scores, n_elites=2)

        # step 8: update prob_greedy
        prob_greedy = update_prob_greedy(min_prob_greedy=0.4, epoch=i, max_epochs=n_gen)

    return best_chromosome, fitness_df, history
