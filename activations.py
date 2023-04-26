import numpy as np

class ReLU():
    def __init__(self):
        pass

    def forward(self, inp):
        return np.maximum(0, inp)
        
    def derivative(self, inp):
        return inp > 0

class Softmax():
    def __init__(self):
        pass

    def forward(self, inp):
        A = np.exp(inp) / sum(np.exp(inp))
        return A
    def derivative(self, inp):
        s = self.forward(inp)
        return s * (1 - s)
