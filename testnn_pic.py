from network import Network 
from dense import Dense
from tanh import Tanh
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

prep_x_train=[]
prep_y_train=[]
prep_x_test=[]
prep_y_test=[]
images=[]
for x, y in zip(x_train, y_train):
    prep_x_train.append(x.reshape(-1, 1))   # (784, 1)
    prep_y_train.append(y.reshape(-1, 1))

for x, y in zip(x_test, y_test):
    images.append(x)
    prep_x_test.append(x.reshape(-1, 1))   # (784, 1)
    prep_y_test.append(y)


network=Network()
network.add_layer(Dense(784, 128))
network.add_layer(Tanh())
network.add_layer(Dense(128, 10))
network.add_layer(Tanh())


epochs = 10
learning_rate = 0.1
res=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

network.train(prep_x_train, prep_y_train,epochs, learning_rate)
network.run(data=prep_x_test, expected_res=prep_y_test, options=res, images=images)