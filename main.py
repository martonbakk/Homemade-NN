from dense import Dense
from tanh import Tanh
from losses import mse, mse_prime
import numpy as np

X= np.reshape([[0,0], [0, 1], [1, 0], [1, 1]], (4,2,1))
Y= np.reshape([[0], [1], [1], [0]], (4,1,1))

print(X)
print(Y)

network=[
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

epochs=500
learning_rate=0.1

for e in range(epochs):
    error=0
    for x,y in zip(X,Y):
        output=x 
        for layer in network:
            output=layer.forward(output)
        
        error+=mse(y, output)

        grad=mse_prime(y, output)
        for layer in reversed(network):
            grad=layer.backward(grad, learning_rate)

    error/=len(x)
    print('%d %d, error=%f' %(e+1, epochs, error))


res=[False, True]

inputs = [X[0], X[1], X[2], X[3]]
for i, input_data in enumerate(inputs):
    output = input_data
    for layer in network:
        output = layer.forward(output)
    print(f"A hálózat kimenete {i+1}: {output}")