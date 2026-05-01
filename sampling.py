import pandas as pd
import DataIntegration
import random

from Instance import Instance

def course_sampling(num_dpt=4, num_courses=50,seed=111):
    random.seed(seed)
    dpt=random.sample(department_list,k=num_dpt)
    df=df_groups[df_groups['Department'].isin(dpt)]
    courses=random.sample(df['Code'].unique().tolist(),k=num_courses)
    df=df_groups[df_groups['Code'].isin(courses)]
    return df

def classroom_sampling(df_course, num_rooms=10,seed=111):
    obj=df_course['NUMSTU'].tolist()
    smallest=min(obj)
    largest=max(obj)
    candidate_pool=df_rooms[df_rooms['Capacity']>=smallest]
    sampled=candidate_pool.sample(num_rooms,random_state=seed)
    if sampled['Capacity'].max()<largest:
        sampled=sampled.drop(sampled.sample(n=1,random_state=seed+1).index)
        big_room_pool=df_rooms[(df_rooms['Capacity']>=largest)]
        new_room=big_room_pool.sample(n=1,random_state=seed+2)
        df=pd.concat([sampled,new_room])
        return df
    else:
        return sampled

def generating_random_schedule(df_course, df_classroom, seed=111):
    random.seed(seed)
    num_rooms = len(df_classroom)
    num_courses = len(df_course)

    ts = [random.randint(1, num_ts) for _ in range(num_courses)]
    rm = [random.randint(0, num_rooms - 1) for _ in range(num_courses)]
    course_cg = df_course['cg'].tolist()
    room_idx = df_classroom.index
    capacity = df_classroom['Capacity'].tolist()
    prfs = df_course['P-Index'].tolist()
    dl = df_course['dl'].tolist()
    cpct = []
    nmst = df_course['NUMSTU'].tolist()
    for i in range(num_courses):
        cpct = cpct + [capacity[rm[i]]]
        rm[i] = room_idx[rm[i]]

    df = pd.DataFrame(
        {'course index': df_course.index, 'code group': course_cg, 'dl': dl, 'timeslot': ts, 'classroom index': rm,
         'professor': prfs, 'capacity': cpct, 'numstu': nmst})

    return df




if __name__ == '__main__':

    df_groups=DataIntegration.groups
    df_groups['Course_Level']=df_groups['Code'].str[2:3]
    df_groups['Index']=df_groups.index
    df_groups['cg']=df_groups['Code']+'-'+df_groups['Group']
    df_groups['dl']=df_groups['Department']+'-'+df_groups['Course_Level']

    df_rooms=DataIntegration.teaching_rooms
    df_rooms['R-Index']=df_rooms.index
    # df_roomss=df_rooms.sort_values(by='Capacity')


    df_timetable=DataIntegration.timetable_by_course_code
    df_timetable['Group']=df_timetable['Group'].astype(str)
    df_timetable.loc[df_timetable['Type'] == 'LECTURE', 'Group'] = 'L'
    df_timetable.loc[df_timetable['Type'] == 'WORKSHOP', 'Group'] = 'W'

    df_timetable['cg']=df_timetable['Code']+'-'+df_timetable['Group']

    df_staff=DataIntegration.staff_df
    df_staff=df_staff.reset_index(drop=True)
    department_list=df_groups['Department'].unique().tolist()

    num_ts=25
    staff_list=df_staff['Staff'].unique().tolist()


    professor_to_index={name:i for i,name in enumerate(staff_list)}
    df_staff['Index']=df_staff['Staff'].map(professor_to_index)
    df_timetable['P-Index']=df_timetable['Staff'].map(professor_to_index)

    cg_to_prof=(df_timetable[['cg','P-Index']].dropna(subset=['cg','P-Index']).drop_duplicates().groupby('cg',as_index=False).sample(n=1,random_state=111))

    df_groups=df_groups.merge(cg_to_prof,on='cg',how='left',validate='many_to_one')
    cg_to_prof=df_groups.set_index('cg')['P-Index'].to_dict()

    # course_to_professor=defaultdict(set)
    # professor_to_course=defaultdict(set)
    #
    # for i in range(len(df_staff)):
    #     course_to_professor[df_staff['Code'][i]].add(professor_to_index[df_staff['Staff'][i]])
    #     professor_to_course[professor_to_index[df_staff['Staff'][i]]].add(df_staff['Code'][i])

    # a=course_sampling()
    # b=classroom_sampling(a)
    # d=a['P-Index'].unique().tolist()
    # c=generating_random_schedule(a,b)

    num_dept=10
    num_courses=400
    num_rooms=50

    num_sample=200

    random.seed(457)
    seeds=random.sample(range(1,10000000), num_sample)
    all_courses_sampling=[]
    all_rooms_sampling=[]
    all_professors=[]
    all_poor_schedule=[]
    for i in range(len(seeds)):
        df_course=course_sampling(num_dept,num_courses,seeds[i])
        df_room=classroom_sampling(df_course,num_rooms,seeds[i]+1)
        prof=df_course['P-Index'].unique().tolist()
        prof=pd.DataFrame({'sample_id':i,'P-Index':prof})
        df_poor_schedule=generating_random_schedule(df_course, df_room,seed=seeds[i]+2)

        df_course['sample_id']=i
        df_room['sample_id']=i
        df_poor_schedule['sample_id']=i

        all_courses_sampling.append(df_course)
        all_rooms_sampling.append(df_room)
        all_professors.append(prof)
        all_poor_schedule.append(df_poor_schedule)

    df_all_courses=pd.concat(all_courses_sampling,ignore_index=True)
    df_all_rooms=pd.concat(all_rooms_sampling,ignore_index=True)
    df_all_professors=pd.concat(all_professors,ignore_index=True)
    df_all_poor_schedule=pd.concat(all_poor_schedule,ignore_index=True)

    df_all_courses.to_csv('Sample/courses_sampling_' + str(num_courses) + '.csv',index=False)
    df_all_rooms.to_csv('Sample/rooms_sampling_' + str(num_rooms) + '.csv',index=False)
    df_all_professors.to_csv('Sample/professors_sampling_' + str(num_courses) + '.csv',index=False)
    df_all_poor_schedule.to_csv('Sample/poor_schedule_generating_' + str(num_courses) + '.csv',index=False)



    all_opt_solution=[]
    for i in range(len(seeds)):
        print('Sample:', i)
        ist=Instance(all_courses_sampling[i], all_rooms_sampling[i],25,all_professors[i]['P-Index'].tolist())
        ist.generate_Gurobi_model()
        ist.solve_Gurobi()
        df=ist.df_solution()
        if df is not None:
            df['sample_id']=i
            all_opt_solution.append(df)
    df_all_opt_solution=pd.concat(all_opt_solution,ignore_index=True)
    df_all_opt_solution.to_csv('Sample/optimized_schedule_' + str(num_courses) + '.csv',index=False)

    
    

    
    
    
    
    
