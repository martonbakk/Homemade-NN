from dense import Dense
from tanh import Tanh
from losses import mse, mse_prime
import numpy as np
from network import Network

prep_x_train= np.reshape([[0,0], [0, 1], [1, 0], [1, 1]], (4,2,1))
prep_y_train= np.reshape([[0], [1], [1], [0]], (4,1,1))


network=[
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

network=Network()
network.add_layer(Dense(2, 3))
network.add_layer(Tanh())
network.add_layer(Dense(3, 1))
network.add_layer(Tanh())

epochs=500
learning_rate=0.1
res=[False, True]

network.train(prep_x_train, prep_y_train,epochs, learning_rate)
network.run(data=prep_x_train, expected_res=prep_y_train, options=res)