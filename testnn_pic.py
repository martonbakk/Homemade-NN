from network import Network 
from dense import Dense
from tanh import Tanh
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

#DATA
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#TRAINING AND TESTING DATA (PIC)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
#TRAINING AND TESTING DATA (LABELS)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
#PREPARING THE DATA 
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

####### NEURAL NETWORK ######
network=Network()
network.add_layer(Dense(784, 128))
network.add_layer(Tanh())
network.add_layer(Dense(128, 10))
network.add_layer(Tanh())
#NEURAL NETWORK CONFIGS
epochs = 10
learning_rate = 0.1
res=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#NEURAL NETWORK TRAINING AND CHECK THE PRED
network.train(prep_x_train, prep_y_train,epochs, learning_rate)
network.run(data=prep_x_test, expected_res=prep_y_test, options=res, images=images)