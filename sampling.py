import numpy as np
import pandas as pd
import DataIntegration
import random
import networkx as nx

from collections import defaultdict
from Instance import Instance
from Graph import Graph
#from node2vec import Node2Vec

df1=DataIntegration.groups
df1['Course_Level']=df1['Code'].str[2:3]
df1['Index']=df1.index
df1['cg']=df1['Code']+'-'+df1['Group']
df1['dl']=df1['Department']+'-'+df1['Course_Level']

df2=DataIntegration.teaching_rooms
df2['R-Index']=df2.index
df2s=df2.sort_values(by='Capacity')


df3=DataIntegration.timetable_by_course_code
df3['Group']=df3['Group'].astype(str)
df3.loc[df3['Type'] == 'LECTURE', 'Group'] = 'L'
df3.loc[df3['Type'] == 'WORKSHOP', 'Group'] = 'W'

df3['cg']=df3['Code']+'-'+df3['Group']

df4=DataIntegration.staff_df
df4=df4.reset_index(drop=True)
department_list=df1['Department'].unique().tolist()

num_ts=25
staff_list=df4['Staff'].unique().tolist()


professor_to_index={name:i for i,name in enumerate(staff_list)}
df4['Index']=df4['Staff'].map(professor_to_index)
df3['P-Index']=df3['Staff'].map(professor_to_index)


cg_to_prof=(df3[['cg','P-Index']].dropna(subset=['cg','P-Index']).drop_duplicates().groupby('cg',as_index=False).sample(n=1,random_state=111))

df1=df1.merge(cg_to_prof,on='cg',how='left',validate='many_to_one')
cg_to_prof=df1.set_index('cg')['P-Index'].to_dict()


course_to_professor=defaultdict(set)
professor_to_course=defaultdict(set)

for i in range(len(df4)):
    course_to_professor[df4['Code'][i]].add(professor_to_index[df4['Staff'][i]])
    professor_to_course[professor_to_index[df4['Staff'][i]]].add(df4['Code'][i])






def course_sampling(num_dpt=3,num_courses=50,seed=111):
    random.seed(seed)
    dpt=random.sample(department_list,k=num_dpt)
    df=df1[df1['Department'].isin(dpt)]
    df=df.sample(n=num_courses,random_state=seed)
    return df
a=course_sampling()

def classroom_sampling(df_course,num_rooms=10,seed=111):
    obj=df_course['NUMSTU'].tolist()
    smallest=min(obj)
    largest=max(obj)
    candidate_pool=df2[df2['Capacity']>=smallest]
    sampled=candidate_pool.sample(num_rooms,random_state=seed)
    if sampled['Capacity'].max()<largest:
        sampled=sampled.drop(sampled.sample(n=1,random_state=seed+1).index)
        big_room_pool=df2[(df2['Capacity']>=largest)]
        new_room=big_room_pool.sample(n=1,random_state=seed+2)
        df=pd.concat([sampled,new_room])
        return df
    else:
        return sampled
b=classroom_sampling(a)

# def professor_sampling(df_course,seed=111):
#     random.seed(seed)
#     pf_list=[]
#     for i in range(len(df_course)):
#         qualified_pf=list(course_to_professor[df_course.iloc[i]['Code']])
#         nm=math.ceil(len(qualified_pf)/2)
#         qualified_pf=random.sample(qualified_pf,k=nm)
#         pf_list=pf_list+qualified_pf
#     pf_list=set(pf_list)
#     return list(pf_list)

# d=professor_sampling(a)
d=a['P-Index'].unique().tolist()


def generating_trash_schedule(df_course,df_classroom,seed=111):
    
    random.seed(seed)

    num_rooms=len(df_classroom)
    num_courses=len(df_course)
    ts=[random.randint(1,num_ts) for _ in range(num_courses)]
    rm=[random.randint(0,num_rooms-1) for _ in range(num_courses)]
    course_cg=df_course['cg'].tolist()
    room_idx=df_classroom.index
    capacity=df_classroom['Capacity'].tolist()
    prfs=df_course['P-Index'].tolist()
    dl=df_course['dl'].tolist()
    cpct=[]
    nmst=df_course['NUMSTU'].tolist()
    for i in range(num_courses):
        cpct=cpct+[capacity[rm[i]]]
        rm[i]=room_idx[rm[i]]
        
        
    df=pd.DataFrame({'course index':df_course.index,'code group':course_cg,'dl':dl,'timeslot':ts,'classroom index':rm,'professor':prfs,'capacity':cpct,'numstu':nmst})
    
    return df
c=generating_trash_schedule(a,b)

#%%



ist=Instance(a, b, 25, d)
gr=Graph(ist,c)
gr._check_constraints()
gr._build_features()



#%%
    
random.seed(457)
seeds=random.sample(range(1,10000000), 1000)
all_courses_sampling=[]
all_rooms_sampling=[]
all_professors=[]
all_poor_schedule=[]
for i in range(len(seeds)):
    df_course=course_sampling(3,50,seeds[i])
    df_room=classroom_sampling(df_course,10,seeds[i]+1)
    prof=df_course['P-Index'].unique().tolist()
    prof=pd.DataFrame({'sample_id':i,'P-Index':prof})
    df_poor_schedule=generating_trash_schedule(df_course, df_room,seed=seeds[i]+2)
    
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

    
    
df_all_courses.to_csv('courses_sampling.csv',index=False)
df_all_rooms.to_csv('rooms_sampling.csv',index=False)
df_all_professors.to_csv('professors_sampling.csv',index=False)
df_all_poor_schedule.to_csv('poor_schedule_generating.csv',index=False)
    
    
    
#%%
all_opt_solution=[]
for i in range(len(seeds)):
    ist=Instance(all_courses_sampling[i], all_rooms_sampling[i],25,all_professors[i]['P-Index'].tolist())
    ist.generate_Gurobi_model()
    ist.solve_Gurobi()
    df=ist.df_solution()
    if df is not None:
        df['sample_id']=i
        all_opt_solution.append(df)
df_all_opt_solution=pd.concat(all_opt_solution,ignore_index=True)
df_all_opt_solution.to_csv('optimized_schedule.csv',index=False)

    
    

    
    
    
    
    
