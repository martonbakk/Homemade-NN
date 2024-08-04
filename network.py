import numpy as np
from dense import Dense
from tanh import Tanh
from losses import mse, mse_prime
import matplotlib.pyplot as plt


class Network:
    def __init__(self):
        self.layers=[]
        
    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, inputs, expected_outputs, epochs, learning_rate):
        state=1
        for e in range(epochs):
            error=0
            for output, exoutput in zip(inputs, expected_outputs):
                state+=1
                for layer in self.layers:
                    output = layer.forward(output)
                error += mse(exoutput, output)
                gradre = mse_prime(exoutput, output)
                grad=gradre[:,0].reshape(10, 1)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
                print(f"{state/(len(inputs)*epochs)*100}")

            error /= len(inputs)
            print('%d/%d, error=%f' % (e+1, epochs, error))
    
    def run(self, data, expected_res,options, images):
        print(len(data[:5]))
        for i, input_data in enumerate(data[:5]):
            output = input_data
            for layer in self.layers:
                output = layer.forward(output)
            print(f"A hálózat kimenete {i+1}: {options[np.argmax(output)]} helyes: {options[np.argmax(expected_res[i])]}")
            plt.imshow(images[i])
            plt.show()
