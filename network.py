#http://neuralnetworksanddeeplearning.com by Michael Nielsen is a reference
#https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ by Matt Mazur is another reference
import layer
import functions as f
import numpy as np #www.numpy.org

class network:
    def __init__(self):
        self.layers = []

    def layer(self, num_nodes, function, input_num = None):
        errvalue = False
        if (input_num == None and self.layers != []):
                innum = len((self.layers[len(self.layers)-1]).nnodes)
        elif (self.layers != [] and  input_num != len((self.layers[len(self.layers)-1]).nnodes)):
            print ("ERR: Input dimension mismatch, layer not created.")
            errvalue = True
        elif (self.layers == [] and input_num == None):
            print ("ERR: No input dimension specified.")
            errvalue = True
        elif (self.layers == [] and input_num != None):
                innum = input_num
        if (errvalue == False):
            self.layers += [layer.layer(innum, num_nodes, function)]
        return errvalue

    def setweights(self, layer, weights, node = None):
        self.layers[layer].setweights(weights, node)
        return True

    def setbias(self, layer, bias, node = None):
        self.layers[layer].setbias(bias, node)
        return True

    def forward(self, inp, keep_vals = False):
        output = [inp]
        count = 0

        while (len(self.layers) != count):
            output += [(self.layers[count]).compute(output[count])]
            count += 1
        if (keep_vals == False):
            return output[count]
        else:
            return output

    def backprop(self, inp, expected, eta):
        out = self.forward(inp, True)
        depth = len(self.layers)
        updated = [[]] * depth
        delta = [[]] * depth
        updated_bias = [[]] * depth
        # Compute deltas, and populate updated with old weights
        delta [depth-1] = np.multiply(np.subtract(np.array(expected), np.array(out[depth])), eval('f.' + ((self.layers[depth-1]).func).__name__ + '_der')(np.array(out[depth])))

        for i in range(depth - 2, -1, -1):
            fder = eval('f.' + ((self.layers[i]).func).__name__ + '_der')
            weights = []
            bias = []
            for k in self.layers[i+1].nnodes:
                weights += [list(k.weights)]
                bias += [float(k.bias)]
            for j in range(len(self.layers[i].nnodes)):
                delta[i] += [np.dot(np.transpose(np.array(weights))[j], delta[i+1])*fder(out[i+1][j])]
            delta[i] = np.array(delta[i])
            updated[i+1] = weights
            updated_bias[i+1] = bias
            
        for k in self.layers[0].nnodes:
            updated[0] += [list(k.weights)]
            updated_bias[0] += [float(k.bias)]

        # Compute new weights

        for i in range(depth):
            for j in range(len(updated[i])):
                for k in range(len(updated[i][j])):
                    updated[i][j][k] += (eta)*(delta[i][j])*(out[i][k])
                updated_bias[i][j] += (eta)*(delta[i][j])
       
        for i in range(depth):
            self.setweights(i, updated[i])
            self.setbias(i, updated_bias[i])
