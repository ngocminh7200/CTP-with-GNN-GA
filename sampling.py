import pandas as pd
import DataIntegration
import random

from Instance import Instance
# professor and staff mean the same in this file

# Sample [by default 50] different course code from [by default 4] different departments, with seed [by default 111]
# A course code may correspond to mulitple groups, grab them all
# Return a DataFrame that contains all rows of courses with sellected code(, excerpted from df_groups)
def course_sampling(num_dpt=4, num_courses=50,seed=111):
    random.seed(seed)   # set seed
    dpt=random.sample(department_list,k=num_dpt)   # sample departments
    df=df_groups[df_groups['Department'].isin(dpt)]   # store [all rows of courses from [df_groups](the full DataFrame of courses we are interested in) whose department is selected] in df
    courses=random.sample(df['Code'].unique().tolist(),k=num_courses)  # obtain a list of unique course code in df, sample [num_courses] code from it, and store in [courses]
    df=df_groups[df_groups['Code'].isin(courses)]  # obtain all rows of courses whose course code is in the selected list [courses]
    return df


# sample [by default 10] classrooms with seed [by default 111] according to a give list of courses,
# so that the largest classroom can accommodate the largest class size
# and the smallest classroom can accommodate the smallest class size
def classroom_sampling(df_course, num_rooms=10,seed=111):
    obj=df_course['NUMSTU'].tolist()  # obtain a list of class size from given DataFrame [df_course] and store in [obj]
    smallest=min(obj)  # smallest class size
    largest=max(obj)  # largest class size
    candidate_pool=df_rooms[df_rooms['Capacity']>=smallest]  # a DataFrame excerpted from [df_rooms](the full DataFrame of rooms), that contains all candidate rooms that can accommodate the smallest class size
    sampled=candidate_pool.sample(num_rooms,random_state=seed)  #  sample [num_rooms] rows of rooms from [candidate_pool]
    if sampled['Capacity'].max()<largest:  # if the largest room we have selected is too small for the largest class size
        sampled=sampled.drop(sampled.sample(n=1,random_state=seed+1).index)  # delete a random row of room from our selected rooms [sampled]
        big_room_pool=df_rooms[(df_rooms['Capacity']>=largest)]  # obtain a DataFrame excerpted from [df_rooms], that contains all rows of rooms that can accommodate the largest class size
        new_room=big_room_pool.sample(n=1,random_state=seed+2)  # sample a new room row from [big_room_pool] (as a compensate for the room we deleted before)
        df=pd.concat([sampled,new_room])  # combine the new room row with the remaining selected room rows.
        return df
    else:  
        return sampled  # return selected rows of rooms, if it already includes a large enough room


# now we have a DataFrame of selected courses, and a DataFrame of selected classrooms
# it's time to start generating random (not necessarily feasible) schedules
# the following function focuses on assigning random rooms(from the selected DataFrame) and timeslots to each selected courses
# the output DataFrame of schedule will later be processed into graph and learned by GNN, as well as its corresponding optimized schedule
def generating_random_schedule(df_course, df_classroom, seed=111):
    random.seed(seed)
    num_rooms = len(df_classroom)  #  number of selected rooms
    num_courses = len(df_course)  #  number of selected courses

    ts = [random.randint(1, num_ts) for _ in range(num_courses)]  #  generate a timeslot for each course, store as a list [ts], timeslot is randomly selected from integers in [1,num_ts(25)] 
    rm = [random.randint(0, num_rooms - 1) for _ in range(num_courses)]  #  generate a room for each course, store as a list [rm]. rooms are chosen from [df_classroom]. rooms are indicated by 0,1,2,3... according to their position in [df_classroom]
    course_cg = df_course['cg'].tolist()  # 'cg' short for 'course group', is a combined column of 'code' and 'group', we stored it to a list [course_cg]
    room_idx = df_classroom.index  # the true row index for selected rooms, indicates their position in the full DataFrame df_rooms
    capacity = df_classroom['Capacity'].tolist()  # a list of room capacity of selected rooms
    prfs = df_course['P-Index'].tolist()  # 'P-Index' is an index for professors, indicating their true position in [df_staff['Staff'].unique().tolist()]
    dl = df_course['dl'].tolist()  # a list of column 'dl' in selected courses DataFrame, short for 'department level', is a combination of 'department' and 'course level'
    cpct = []  
    nmst = df_course['NUMSTU'].tolist()  # a list of column 'NUMSTU' in selected courses DataFrame
    for i in range(num_courses):
        cpct = cpct + [capacity[rm[i]]]  # as we assigned random rooms to each course, by using their position in the selected rooms DataFrame, we now extract their corresponding capacity list
        rm[i] = room_idx[rm[i]]  # and replace variable rm with a list of their true room index

    df = pd.DataFrame(
        {'course index': df_course.index, 'code group': course_cg, 'dl': dl, 'timeslot': ts, 'classroom index': rm,
         'professor': prfs, 'capacity': cpct, 'numstu': nmst})  # stick the schedule and every related information together into a DataFrame for future use
    return df




if __name__ == '__main__':  # the following code will be run if and only if this file is executed directly, as part of them are computation heavy sampling work

    # some further data processing 
    df_groups=DataIntegration.groups  # import DataFrame variable 'groups' from DataIntegration.py
    df_groups['Course_Level']=df_groups['Code'].str[2:3]  # add a new column to indicate course level. e.g. ST457 has a course level 4
    df_groups['Index']=df_groups.index  # add a new column 'Index', according to the row index. 'Index' is a reliable way of indicating a specific course. Different courses have different 'Index'
    df_groups['cg']=df_groups['Code']+'-'+df_groups['Group']  # Combine 'Code' and 'Group' to produce a new column 'cg'
    df_groups['dl']=df_groups['Department']+'-'+df_groups['Course_Level']  # combine 'Department' and 'Course_Level' to produce a new column 'dl'

    df_rooms=DataIntegration.teaching_rooms  # import DataFrame variable 'teaching_rooms' from DataIntegration.py
    df_rooms['R-Index']=df_rooms.index  # add a new column 'R-Index', according to the row index of [df_rooms]. Different rooms have different 'R-Index'

    df_timetable=DataIntegration.timetable_by_course_code  # import DataFrame variable 'time_table_by_course_code' from DataIntegration.py. This is the raw large course dataset, which includes everything. 
                                                           # we import it here mainly to extract its 'Staff' column and assign it to corresponding group
    df_timetable['Group']=df_timetable['Group'].astype(str)  # turn them into strings because this column originally uses numbers to indicate groups
    df_timetable.loc[df_timetable['Type'] == 'LECTURE', 'Group'] = 'L'  # lectures transformed into group 'L' instead of '1'
    df_timetable.loc[df_timetable['Type'] == 'WORKSHOP', 'Group'] = 'W'  # workshop transformed into group 'W' instead of '1'

    df_timetable['cg']=df_timetable['Code']+'-'+df_timetable['Group']  # added a new column 'cg' the same way as above, for the future sake of assigning professors

    df_staff=DataIntegration.staff_df  # import DataFrame variable 'staff_df' from DataIntegration.py. This is a DataFrame includes which professor can teach which course code, but without group information
    df_staff=df_staff.reset_index(drop=True)  # I think this line is no longer needed. It attempts to reset the row index of this DataFrame into 0,1,2,3... though
    department_list=df_groups['Department'].unique().tolist()  # I think this line is no longer needed. It obtains a unique list of dapartment names. 

    num_ts=25  # number of timeslots. will be used in the random schedule generating function defined above
    staff_list=df_staff['Staff'].unique().tolist()  # a unique list of staff, according to their order in df_staff. this list is future used to index staffs


    professor_to_index={name:i for i,name in enumerate(staff_list)}  # a dictionary. profesoor_to_index['professor_name'] shall return their index. 'professor_name' is the indiscribable string in the 'Staff' column in df_staff
    df_staff['Index']=df_staff['Staff'].map(professor_to_index)  # add a new column 'Index' to [df_staff] accoding to the above dictionary 
    df_timetable['P-Index']=df_timetable['Staff'].map(professor_to_index)  # add a new column 'P-Index' to [df_timetable] according the true index of the corresponding professor

    cg_to_prof=(df_timetable[['cg','P-Index']].dropna(subset=['cg','P-Index']).drop_duplicates().groupby('cg',as_index=False).sample(n=1,random_state=111))  # attempts to find a corresponding professor for each course group according to the full timetable they gave us
                                                                                                                                                             # however, there are outliers that are taught by different professor in different weeks, 
                                                                                                                                                             # in these cases, we will randomly assign one of the professors to this group
                                                                                                                                                             # if you have any concerns, please consult the author of DataIntegration.py

    df_groups=df_groups.merge(cg_to_prof,on='cg',how='left',validate='many_to_one')  # The above DataFrame has assigned a professor to each group, this line adds a corresponding 'P-Index' for each row by doing left join 
    cg_to_prof=df_groups.set_index('cg')['P-Index'].to_dict()  # and don't waste the cg_to_prof DataFrame we created, make it into a dictionary just in case

    # run the following four commented lines will give you a sample of the the result of the three sampling function defined above
    # a=course_sampling()
    # b=classroom_sampling(a)
    # d=a['P-Index'].unique().tolist()
    # c=generating_random_schedule(a,b)

    # starting from this line, is the sampling part
    num_dept=4  # number of departments you would like to sample
    num_courses=50  # number of course code within the above departments you would like to sample
    num_rooms=10  # number of classrooms you would like to sample

    num_sample=1000  # number of cases you would like to sample
    random.seed(457)  # random seed. you may change as you wish, but please keep record of it if it's not 457 for re-producing reasons.
    
    seeds=random.sample(range(1,10000000), num_sample)  # generate a list of random seeds
    all_courses_sampling=[] 
    all_rooms_sampling=[]
    all_professors=[]
    all_poor_schedule=[]

    # generating courses, classrooms, timetables
    for i in range(len(seeds)):  # a different random seed for each loop
        df_course=course_sampling(num_dept,num_courses,seeds[i])  # call function defined above, generating courses
        df_room=classroom_sampling(df_course,num_rooms,seeds[i]+1)  # call function defined above, generating rooms
        prof=df_course['P-Index'].unique().tolist()  # export a unique list of professor index from the selected courses DataFrame
        prof=pd.DataFrame({'sample_id':i,'P-Index':prof}) # store this list of professor index into a DataFrame, accompanied with sample_id, which is noted as the loop index
        df_poor_schedule=generating_random_schedule(df_course, df_room,seed=seeds[i]+2)  # call function defined above, generating random schedules

        df_course['sample_id']=i  # add a new column to [df_course] which indicates which sample group it is in
        df_room['sample_id']=i  # add a new column to [df_room] which indicates which sample group it is in
        df_poor_schedule['sample_id']=i  # add a new column to [df_poor_schedule] which indicates which sample group it is in

        # the following four lines stored courses, rooms, professors, random schedule generated in this loop into four variables respectively
        all_courses_sampling.append(df_course)  
        all_rooms_sampling.append(df_room)
        all_professors.append(prof)
        all_poor_schedule.append(df_poor_schedule)

    # all sample generation has done by now, the following four lines convert the above four [list of DataFrames] into four DataFrames
    df_all_courses=pd.concat(all_courses_sampling,ignore_index=True)
    df_all_rooms=pd.concat(all_rooms_sampling,ignore_index=True)
    df_all_professors=pd.concat(all_professors,ignore_index=True)
    df_all_poor_schedule=pd.concat(all_poor_schedule,ignore_index=True)

    # and then output into .csv files stored in the folder 'Sample'
    df_all_courses.to_csv('Sample/courses_sampling_' + str(num_courses) + '.csv',index=False)
    df_all_rooms.to_csv('Sample/rooms_sampling_' + str(num_rooms) + '.csv',index=False)
    df_all_professors.to_csv('Sample/professors_sampling_' + str(num_courses) + '.csv',index=False)
    df_all_poor_schedule.to_csv('Sample/poor_schedule_generating_' + str(num_courses) + '.csv',index=False)

    
    # the following part focuses on using Gurobi solver to solve for optimized solution according to each cases generated above
    all_opt_solution=[]  # create an empty string for appending loop outputs
    red='\033[91m'  # print this, and text after it will become red
    RESET='\033[0m'  # print this, and text after it will change back to their original color
    for i in range(len(seeds)):  # a different seed for each loop
        ist=Instance(all_courses_sampling[i], all_rooms_sampling[i],25,all_professors[i]['P-Index'].tolist())  # Instance is a new class defined in Instance.py
        ist.generate_Gurobi_model() # the instance function, to summarize, it will build the gurobi model by adding constraints, according to that file, the model is stored in ist.model
        ist.solve_Gurobi()  # this will solve the model built above
        df=ist.df_solution()  # returns the solution as a DataFrame. It will return 'None' if hard constraints cannot be satisfied (i.e. model infeasible)
        if df is not None:  # so only solutions will be recorded, with sample index accompanied. infeasible samples will be skipped
            df['sample_id']=i  # add a column that records the sample index (i.e. which sample does this solution corresponds to)
            all_opt_solution.append(df)  # append the solution DataFrame to the list.
        print(f'{red}sample:{i}{RESET}')  # have you seen red progression text?
    df_all_opt_solution=pd.concat(all_opt_solution,ignore_index=True)  # concrete the [list of solution DataFrame] into one DataFrame
    df_all_opt_solution.to_csv('Sample/optimized_schedule_' + str(num_courses) + '.csv',index=False)  # convert the DataFrame into .csv and store in the folder 'Sample'

    
    

    
    
    
    
    
