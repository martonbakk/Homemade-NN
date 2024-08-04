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

    def train(self, inputs, expected_outputs, epochs):
        for e in range(epochs):
            error=0
            for output, exoutput in zip(inputs, expected_outputs):
                 for layer in self.layers:
                    output = layer.forward(output)
            error /= len(inputs)
            print('%d/%d, error=%f' % (e+1, epochs, error))
