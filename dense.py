from layer import Layer
import numpy as np

#Y: output matrix
#W: weight matrix
#X: input matrix

class Dense(Layer):
    def __init__(self, input_size, output_size) -> None:        #Size of the layer i/o
        self.weights=np.random.randn(output_size, input_size)   #Weights  W matrix init randomly
        self.bias=np.random.randn(output_size, 1)               #Bias     B matrix init randomly
    
    def forward(self, input):                               #Get an input X matrix
        self.input=input                                    #We set the input 
        return np.dot(self.weights, self.input)+self.bias   #Matrix multiplication --W*X+B--
    
    def backward(self, output_gradient, learning_rate):         #output gradient= del(E)/del(Y)   
        weights_gradient=np.dot(output_gradient, self.input.T)  #calculate the weights grad
        self.weights-=learning_rate*weights_gradient            #adjustment
        self.bias-=learning_rate*output_gradient                #adjustmen
        return np.dot(self.weights.T, output_gradient)          #send to the before layer