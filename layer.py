import numpy as np
import node

class layer:
    def __init__(self, in_num, n_nodes, func):
        self.input = in_num
        self.func = func
        self.nnodes = []
        for i in range(n_nodes):
            self.nnodes += [node.node(b = (2*np.random.random_sample())-1, input_num = in_num, w = (2*np.random.random_sample(in_num))-1, func = func)]

    def compute(self, inp):
        accum =[]
        if (len(inp) != self.input):
            return -1
        else:
            for i in self.nnodes:
                accum += [i.compute(inp)]
        return accum

    def setweights(self, inlist, node = None):
        if (node == None):
            for i in range(len(self.nnodes)):
                self.nnodes[i].setweights(inlist[i])
        else:
            self.nnodes[node].setweights(inlist)

    def setbias(self, inlist, node = None):
        if (node == None):
            for i in range(len(self.nnodes)):
                self.nnodes[i].setbias(inlist[i])
        else:
            self.nnodes[node].setbias(inlist)
