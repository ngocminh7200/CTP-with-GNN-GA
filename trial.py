
import numpy as np
from sklearn.neural_network import MLPClassifier
import networkx as nx
import matplotlib.pyplot as plt

# a=np.random.rand(2000,40)
# train_x=a

# b=np.random.randint(0,45,size=(2000,1))
# b=b.ravel()
# train_y=b
# professor=np.random.randint(0,100,size=(2000,1)).ravel()
# timeslots=np.random.randint(0,45,size=(2000,1)).ravel()
# classroom=np.random.randint(0,200,size=(2000,1)).ravel()



def mlp(train_x,train_y,dimension_tuple=(128,64)):
    
    clf=MLPClassifier(
        hidden_layer_sizes=dimension_tuple,
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        random_state=111
    )
    
    return clf.fit(train_x,train_y)

fit_professor=mlp(train_x,professor)
fit_timeslots=mlp(train_x,timeslots)
fit_classroom=mlp(train_x,classroom)


def predict_new_better_schedule(encoded_schedule):
    professor=fit_professor.predict(encoded_schedule)
    timeslots=fit_timeslots.predict(encoded_schedule)
    classroom=fit_classroom.predict(encoded_schedule)
    
    return professor,timeslots,classroom

def generate_graph(professor,timeslots,classroom,num_classes,num_professor,num_timeslots,num_classroom):
    G=nx.Graph()
    for i in range(num_professor):
        G.add_node(f'professor{i+1}', type='professor')
    for i in range(num_timeslots):
        G.add_node(f'timeslots{i+1}', type='timeslots')
    for i in range(num_classroom):
        G.add_node(f'classroom{i+1}', type='classroom')
    for i in range(num_classes):
        G.add_node(f'class{i+1}',node_type='class')
        G.add_edge(f'class{i+1}', f'professor{professor[i]+1}')
        G.add_edge(f'class{i+1}', f'timeslots{timeslots[i]+1}')
        G.add_edge(f'class{i+1}', f'classroom{classroom[i]+1}')
    return G
  


# pred=predict_new_better_schedule(train_x)
# g=generate_graph(professor, timeslots, classroom, 40, 100, 45, 200)

