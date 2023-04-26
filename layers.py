import numpy as np

class Dense():
    def __init__(self, inp_units, outp_units, activation):
        self.weights = np.random.rand(outp_units, inp_units) - 0.5
        self.biases = np.random.rand(outp_units, 1) - 0.5
        self.neuron_count = outp_units
        self.activation = activation
        self.Z = None
        self.A = None
        self.dZ = None
        self.A_prev = None

    def get_neuron_count(self):
        return self.neuron_count

    def forward(self, inp):
        self.A_prev = inp

        self.Z = self.weights.dot(inp) + self.biases
        self.A = self.activation.forward(self.Z)
      
        return self.A
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        
        return one_hot_Y
    def backward(self, Y, learning_rate, m, next_layer=None):
        if next_layer == None:
            one_hot_Y = self.one_hot(Y)
            self.dZ = self.A - one_hot_Y
        else:
            self.dZ = next_layer.weights.T.dot(next_layer.dZ) * self.activation.derivative(self.Z)
        dW = 1 / m * self.dZ.dot(self.A_prev.T)
        db = 1 / m * np.sum(self.dZ)

        # update params
        self.weights = self.weights - learning_rate * dW
        self.biases = self.biases - learning_rate * db 

