import numpy as np
import pandas as pd
import DataIntegration
import random
import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import defaultdict

#from node2vec import Node2Vec

df1=DataIntegration.groups
df1['Course_Level']=df1['Code'].str[2:3]
df2=DataIntegration.teaching_rooms

df4=DataIntegration.staff_df
df4=df4.reset_index(drop=True)
department_list=df1['Department'].unique().tolist()

num_ts=25
staff_list=df4['Staff'].unique().tolist()
professor_to_index={name:i for i,name in enumerate(staff_list)}

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

def classroom_sampling(num_rooms=10,seed=111):
    df=df2.sample(n=num_rooms,random_state=seed)
    return df
b=classroom_sampling()

def professor_sampling(df_course,seed=111):
    random.seed(seed)
    pf_list=[]
    for i in range(len(df_course)):
        qualified_pf=list(course_to_professor[df_course.iloc[i]['Code']])
        nm=math.ceil(len(qualified_pf)/2)
        qualified_pf=random.sample(qualified_pf,k=nm)
        pf_list=pf_list+qualified_pf
    pf_list=set(pf_list)
    return list(pf_list)

d=professor_sampling(a)
        


def generating_trash_schedule(df_course,df_classroom,ls_professor,seed=111):
    random.seed(seed)

    num_rooms=len(df_classroom)
    num_courses=len(df_course)
    ts=[random.randint(1,num_ts) for _ in range(num_courses)]
    rm=[random.randint(0,num_rooms-1) for _ in range(num_courses)]
    course_idx=df_course.index
    room_idx=df_classroom.index
    capacity=df2.loc[room_idx,'Capacity'].tolist()
    
    prfs=[]
    cd=[]
    dpt=[]
    cpct=[]
    nmst=[]
    for i in range(num_courses):
        cpct=cpct+[capacity[rm[i]]]
        rm[i]=room_idx[rm[i]]
        st=df1['NUMSTU'][course_idx[i]]
        nmst=nmst+[st]
        code=df1['Code'][course_idx[i]]
        cd=cd+[code]
        prfs=prfs+[random.choice(list(course_to_professor[code]&set(ls_professor)))]
        department=df1['Department'][course_idx[i]]
        dpt=dpt+[department]
        
    df=pd.DataFrame({'course index':course_idx,'code':cd,'department':dpt,'timeslot':ts,'classroom index':rm,'professor index':prfs,'capacity':cpct,'numstu':nmst})
    
    return df
c=generating_trash_schedule(a,b,d)




def schedule_to_graph(df_schedule):
    G=nx.Graph()
    prfs=df_schedule['professor index'].unique().tolist()
    for i in range(len(prfs)):
        G.add_node(f'professor{prfs[i]}', type='professor')
    for i in range(num_ts):
        G.add_node(f'timeslot{i+1}', type='timeslot')
    rm=df_schedule['classroom index'].unique().tolist()
    for i in range(len(rm)):
        G.add_node(f'classroom{rm[i]}', type='classroom')
    cd=df_schedule['code'].unique().tolist()
    for i in range(len(cd)):
        G.add_node(cd[i], type='code')
    cpct=df_schedule['capacity'].unique().tolist()
    for i in range(len(cpct)):
        G.add_node(f'capacity{cpct[i]}', type='capacity')
    nmst=df_schedule['numstu'].unique().tolist()
    for i in range(len(nmst)):
        G.add_node(f'numstu{nmst[i]}', type='numstu')
    course=df_schedule['course index'].tolist()
    dpt=df_schedule['department'].unique().tolist()
    for i in range(len(dpt)):
        G.add_node(dpt[i], type='department')
    for i in range(len(course)):
        G.add_node(f'course{course[i]}',node_type='course')
        G.add_edge(f'course{course[i]}', f'professor{df_schedule['professor index'][i]}',relationship='is taught by')
        G.add_edge(f'course{course[i]}', df_schedule['code'][i],relationship='has code')
        G.add_edge(f'course{course[i]}', f'classroom{df_schedule['classroom index'][i]}', relationship='take place in')
        G.add_edge(f'course{course[i]}', f'timeslot{df_schedule['timeslot'][i]}',relationship='during time')
        G.add_edge(f'course{course[i]}',df_schedule['department'][i],relationship='belongs to department' )
        G.add_edge(f'course{course[i]}',f'numstu{df_schedule['numstu'][i]}',relationship='has number of students')
    for i in range(len(prfs)):
        G.add_edge(f'professor{prfs[i]}',df4['Code'][prfs[i]],relationship='qualified to teach')
    for i in range(len(rm)):
        G.add_edge(f'classroom{rm[i]}', f'capacity{df_schedule.loc[df_schedule['classroom index']==rm[i],'capacity'].iloc[0]}', relationship='has capacity')
    return G

g=schedule_to_graph(c)
    
    
# def aggregation(graph):
#     node2vec=Node2Vec(graph,dimensions=64,walk_length=30,num_walks=100,workers=28)
#     return node2vec
# aggregation(g)  
    
    
def feed_gurobi(df_course,df_classroom,ls_professor):
    courses=df_course.index
    professors={}
    for i in range(len(ls_professor)):
        professors[ls_professor[i]]=list(professor_to_course[ls_professor[i]])
    rooms={}
    for i in range(len(df_classroom)):
        rooms[df_classroom.iloc[i]['Host Key']]=df_classroom.iloc[i]['Capacity']
    demand={}
    for i in range(len(df_course)):
        demand[df_course.iloc[i]['Code']]=df_course.iloc[i]['NUMSTU']
    programs={department_level:list(index) for department_level,index in df_course.groupby(['Department','Course_Level']).groups.items()}
    return courses,professors,rooms,demand,programs
e=feed_gurobi(a, b, d)
    

        
    
    
    
    
    
    
    
