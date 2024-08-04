import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from dense import Dense
from tanh import Tanh
from losses import mse, mse_prime
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Címkék one-hot kódolása
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

network = [
    Dense(784, 128),
    Tanh(),
    Dense(128, 10),
    Tanh()
]

epochs = 2
learning_rate = 0.1

res=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        output = x.reshape(-1, 1)  # (784, 1)
        expected_out=y.reshape(-1, 1)
        for layer in network:
            output = layer.forward(output)
        error += mse(expected_out, output)
        
        gradre = mse_prime(expected_out, output)
        grad=gradre[:,0].reshape(10, 1)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(x_train)
    print('%d/%d, error=%f' % (e+1, epochs, error))

for i, input_data in enumerate(x_test):
    img=input_data
    output = input_data.reshape(-1, 1)
    for layer in network:
        output = layer.forward(output)
    print(f"A hálózat kimenete {i+1}: {res[np.argmax(output)]} helyes: {res[np.argmax(y_test[i])]}")
    plt.imshow(img)
    plt.show()