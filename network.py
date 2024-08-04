import numpy as np
from dense import Dense
from tanh import Tanh
from losses import mse, mse_prime
import matplotlib.pyplot as plt


class Network:
    def __init__(self):
        self.layers=[]
        pass
    def add_layer(self, layer):
        self.layers.append(layer)

        