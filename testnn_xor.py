from dense import Dense
from tanh import Tanh
import numpy as np
from network import Network

#DATA (HERE THE TRAINING AND THE TESTING DATASETS ARE THE SAME)
prep_x_train= np.reshape([[0,0], [0, 1], [1, 0], [1, 1]], (4,2,1))
prep_y_train= np.reshape([[0], [1], [1], [0]], (4,1,1))
####### NEURAL NETWORK ######
network=Network()
network.add_layer(Dense(2, 3))
network.add_layer(Tanh())
network.add_layer(Dense(3, 1))
network.add_layer(Tanh())
#NEURAL NETWORK CONFIGS
epochs=500
learning_rate=0.1
res=[False, True]
#NEURAL NETWORK TRAINING AND CHECK THE PRED
network.train(prep_x_train, prep_y_train,epochs, learning_rate)
network.run(data=prep_x_train, expected_res=prep_y_train, options=res)