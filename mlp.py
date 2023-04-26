import numpy as np
from activations import ReLU, Softmax
from layers import Dense


class MLP():
    def __init__(self):
        self.layers = []
        self.loss_values = []
        self.acc_values = []

    def categorical_crossentropy_loss(self, y):
        one_hot_y = self.one_hot(y)
        return -np.sum(one_hot_y * np.log(self.layers[-1].A + 1e-8)) / y.size

    def one_hot(self, y):
        one_hot_y = np.zeros((y.size, y.max() + 1))
        one_hot_y[np.arange(y.size), y] = 1
        one_hot_y = one_hot_y.T
        return one_hot_y

    def accuracy(self, predictions, y):
        return np.sum(predictions == y) / y.size

        
    def add_layer(self, neuron_count, activation, inp_shape=None):
        if len(self.layers) == 0 and inp_shape is None:
            raise ValueError("Must defined input shape for first layer")

        if inp_shape is None:
            inp_shape = self.layers[-1].get_neuron_count()

        self.layers.append(Dense(inp_shape, neuron_count, activation))

    def forward(self, X):
        A = X
        for l in self.layers:
            A = l.forward(A)
        
        return A
    
    def backward(self,y, learning_rate):
        m = y.shape[0]
        self.layers[-1].backward(y, learning_rate, m)

        for i in range(len(self.layers)-2, -1, -1):
            self.layers[i].backward(y, learning_rate, m, self.layers[i+1])


    def predict(self, X=None):
        if X is None:
            A = self.layers[-1].A
        else:
            A = self.forward(X)
        return np.argmax(A, 0)

    def fit(self, X, y, learning_rate, epochs):
        print("\n Starting training...\n")
        for i in range(epochs):
            self.forward(X)
            self.backward(y, learning_rate)

            loss = self.categorical_crossentropy_loss(y)
            acc = self.accuracy(self.predict(), y)
            
            self.loss_values.append(loss)
            self.acc_values.append(acc)

            if(i % 10 == 0):
                print(f"Epoch {i}/{epochs} - loss: {loss:.4f} - acc: {acc}")
        print("\n Training is over\n")

    def testing(self, X, y):
        predictions = self.predict(X)
        acc = self.accuracy(predictions, y)
        print(f"\nAccuracy on test data({len(y)} examples): {acc*100}% \n")

    def save_weights(self, filepath):
        weights = [l.weights for l in self.layers]
        biases = [l.biases for l in self.layers]
        np.savez(filepath, weights=weights, biases=biases)

    def load_weights(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        weights = data['weights']
        biases = data['biases']
        for i in range(len(self.layers)):
            self.layers[i].weights = weights[i]
            self.layers[i].biases = biases[i]
