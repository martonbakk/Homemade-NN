from layer  import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prime) -> None:
        self.activation=activation
        self.activation_prime=activation_prime

    def forward(self, input):
        self.input=input
        return self.activation(self.input)
        
    def backward(self, output_gradient, learning_rate):
        #print("output gradient")
        #print(output_gradient)
        #print("self.activation_prime(self.input)")
        #print(self.activation_prime(self.input))
        #print("np.multiply(output_gradient, self.activation_prime(self.input))")
        #print(np.multiply(output_gradient, self.activation_prime(self.input)))
        return np.multiply(output_gradient, self.activation_prime(self.input))