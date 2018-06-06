#Functions gotten from https://en.wikipedia.org/wiki/Activation_function
#Most are probably wrong
#See Neural-Net-2/functions.py for "new" version, may actually be the same
import numpy as np #www.numpy.org

def linear(x):
    return x

def linear_der(x):
    return 1

def relu(inp):
    if (inp < 0):
        return 0
    else:   
        return inp

def relu_der(inp):
    if (inp < 0):
        return 0
    else:
        return 1

def prelu(inp, leak_rate):
    if (inp < 0):
        return ((leak_rate)*inp)
    else:
        return inp

def prelu_der(inp, leak_rate):
    if (inp < 0):
        return leak_rate
    else:
        return 1

def elu(inp, leak_rate):
    if (inp < 0):
        return (leak_rate * (np.exp(inp) - 1))
    else:
        return inp

def elu_der(inp, leak_rate):
    if (inp < 0):
        return (leak_rate * np.exp(inp))
    else:
        return 1

def identity(inp):
    return inp

def identity_der(inp):
    return 1

def arctan(inp):
    return np.arctan(inp)

def arctan_der(inp):
    return (1/(1+(x**2)))

def arctan_inv(inp):
    return np.tan(inp)

def tanh(inp):
    return np.tanh(inp)

def tanh_der(inp):
    return 1 - (np.tanh(inp))**2

def softplus(inp):
    return np.log(1 + np.exp(inp))

def softplus_der(inp):
    return (1/(1 + np.exp(-inp)))

def sigmoid(inp):
    return (1/(1 + np.exp(-inp)))

def sigmoid_der(inp):
    return sigmoid(inp) * (1 - sigmoid(inp))
