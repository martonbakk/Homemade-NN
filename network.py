import numpy as np
from dense import Dense
from tanh import Tanh
from losses import mse, mse_prime
import matplotlib.pyplot as plt
from progressbar import progress_bar

class Network:
    def __init__(self):
        self.layers=[]
        
    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, input):
        output=input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, learning_rate):
        state=1
        for e in range(epochs):
            error=0
            for x, y in zip(x_train, y_train):
                state+=1
                output=self.predict(x)
                error += mse(y, output)

                grad = mse_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
                progress_bar(state, (len(x_train)*epochs), prefix='Haladás:', suffix='Kész', length=50)
            error /= len(x_train)
            print('%d/%d, error=%f' % (e+1, epochs, error))
    
    def run(self, **kwargs):
        for i, input_data in enumerate(kwargs['data']):
            output=self.predict(input_data)
            if 'options' in kwargs and len(output)==1:
                good_answr=kwargs['expected_res'][i]
                op=kwargs['options']
                print(f"A hálózat kimenete {i+1}: {output} helyes: {good_answr}")
            elif 'options' in kwargs and len(output)>0:
                good_answr=kwargs['expected_res'][i]
                op=kwargs['options']
                print(f"A hálózat kimenete {i+1}: {op[np.argmax(output)]} helyes: {np.argmax(good_answr)}")
            if 'images' in kwargs:
                plt.imshow(kwargs['images'][i])
                plt.show()
