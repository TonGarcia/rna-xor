
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import copy
import numpy as np
import pandas as pd
from random import uniform
import matplotlib.pyplot as plt

def MSE(y, Y):
    return np.mean((y-Y)**2)


# In[2]:


class Neuron:
    'Neuron is a superclass that defines the global and default neuron\'s behaviour'
    
    def __init__(self, dimension, act_type, learning_rate):
        """Neuron Constructor.
            #self: means an object scope method, not passed by programmer, but by the interpreter
            @dimension: size of dimension (for XOR it is 2 entries/inputs)
            @output_nodes: an int amount of neurons nodes that process the data & retrieve the result
            @learning_rate: the rate/size of the learning/training process (it's as lower as good)
        """
        
        # Set attrs
        self.dimension = dimension
        self.lrate = learning_rate
        self.act_type = act_type
        
        # Init empty vars that will be updated
        self.out = None
        self.error = None
        
        # Init the aux variables
        self.weights = [uniform(0,1) for x in range(dimension)]
        self.bias = uniform(0, 1)
       
    def activation(self, fx):
        if self.act_type == 'sigmoid':
            return 1/(1 + np.exp(-fx))
        elif self.act_type == 'tanh':
            return 1. - np.square(np.tanh(fx))

    def update(self, inputs):
        """update the weights & it bias
            @input = matrix with the train inputs
        """

        counter = 0
        for i in inputs:
            delta = self.lrate * self.error
            self.weights[counter] -= (delta*i)
            self.bias -= delta
            counter+=1

    def feedforward(self, inputs):
        counter = 0
        sum = self.bias
        for i in inputs:
            sum += i * self.weights[counter]
            counter += 1
        self.out = self.activation(sum)
        
    def backward(self):
        """
            Defined on it children
        """
        pass        


# In[3]:


class HiddenNeuron(Neuron):
    'HiddenNeuron is a child of Neuron that receives feedforward from input layer and the backprop from the Output'
    
    def __init__(self, dim, act_type, lrate=0.2):
        super(HiddenNeuron, self).__init__(dim, act_type, lrate)

    def backward(self, deltas, weights):
        sum = 0
        size = len(deltas)
        for x in range(size):
            sum += deltas[x] * weights[x]
        self.error = self.out * (1 - self.out) * sum


# In[4]:


class OutputNeuron(Neuron):
    'OutputNeuron is a child of Neuron that receives feedforward from hidden layer, backprop to hidden & retrieves it final result'
    def __init__(self, dim, act_type, lrate=0.2):
        super(OutputNeuron, self).__init__(dim, act_type, lrate)

    def backward(self, target):
        self.error = self.out * (1 - self.out) * (self.out - target)


# In[5]:


class Model:
    'Model is the way  Optional class documentation string'
    
    def __init__(self, act_type):
        self.hidden = [HiddenNeuron(2, act_type) for i in range(2)]
        self.output = OutputNeuron(2, act_type)

    def predict(self, input):
        temp = []
        for x in range(2):
            self.hidden[x].feedforward(input)
            temp.append(self.hidden[x].out)
        self.output.feedforward(temp)
        return self.output.out

    def train(self, inputs, targets):
        i = 0
        size = len(inputs)

        if i == size:
            i = 0
        feature = inputs[i]
        temp = []
        for x in range(2):
            self.hidden[x].feedforward(feature)
            temp.append(self.hidden[x].out)
        self.output.feedforward(temp)
        self.output.backward(targets[i])
        deltas = []
        deltas.append(self.output.error)
        weights = []
        weights.append([self.output.weights[0]])
        weights.append([self.output.weights[1]])
        for x in range(2):
            self.hidden[x].backward(deltas, weights[x])
        for x in range(2):
            self.hidden[x].update(feature)
        self.output.update(temp)
        i += 1


# In[6]:


import sys

inputs = [[0,0], [0,1], [1,0], [1,1]]
np_inputs = np.array(inputs.copy())
goals = [0, 1, 1, 0]
ms_loss = []
mt_loss = []

epochs = 100000

ms = Model('sigmoid')
mt = Model('tanh')

# Train
for e in range(epochs):
    ms.train(inputs, goals)
    mt.train(inputs, goals)
    
    # e_ms_loss = MSE(ms.predict(inputs), np_inputs.values)
    # e_mt_loss = MSE(mt.predict(inputs), np_inputs.values)
    #
    # sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] + "% ... Sigmoid loss: " + str(e_ms_loss) + "% ... Tangent loss: " + str(e_mt_loss))
    
    # ms_loss.append(e_ms_loss)
    # mt_loss.append(e_mt_loss)
    

# Plot
plt.plot(ms_loss, label='Sigmoid loss')
plt.plot(ms_loss, label='Hyperbolic Tangent loss')
plt.legend()
plt.ylim(ymax=0.5)


# Predict
for i in inputs:
    newp = 0
    ps = ms.predict(i)
    pt = ms.predict(i)
    newp = 1 if (ps > 0.1) else 0
    newp = 1 if (pt > 0.1) else 0
    
    print('Sigmoid prediction for ' + str(i) + ' => ' + str(newp) + ' => ' + str(ps))
    print('Hyperbolic Tangent for ' + str(i) + ' => ' + str(newp) + ' => ' + str(pt))

