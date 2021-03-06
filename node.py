#Learned this stuff from http://neuralnetworksanddeeplearning.com which is "Neural Networks and Deep Learning" by Michael Nielsen
#See Neural-Net-2/layer.py for replacement
import numpy as np #See www.numpy.org

class node:
    def __init__(self, func, b, input_num, w = []):
        self.function = func
        self.bias = b
        if (w != [] and len(w) == input_num):
            self.weights = w
        else:
            self.weights = (2*np.random.random_sample(input_num))-1
        
    def setweights(self, w):
        if (len(w) == len(self.weights)):
            self.weights = w
    
    def setbias(self, b):
        self.bias = b

    def setfunction(self, func):
        self.function = func

    def compute(self, inputs):
        if (len(inputs) == len(self.weights)):
            return self.function(np.dot(self.weights, inputs) + self.bias)
