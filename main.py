import numpy as np
from preprocess import preprocess_folder
from mlp import MLP
from activations import ReLU, Softmax
from layers import Dense
from visualize import visualize


if __name__ == '__main__':

    X_train, Y_train = preprocess_folder("train_png")
    X_test, Y_test = preprocess_folder("test_png")

    X_test, X_train = X_test.T,  X_train.T

    # if you dont want to use all data 60.000 image for training 
    # X_train = X_train[:, :1500]
    # Y_train = Y_train[:1500]

    # for best min 1400
    epochs = 15
    learning_rate = 0.10

    network = MLP()
    network.add_layer(128, ReLU(), inp_shape=784 )
    network.add_layer(10, Softmax())

    network.fit(X_train, Y_train, learning_rate, epochs)

    # network.save_weights(f'mlp_{epochs}eps.npz')

    visualize(epochs, network)
    # network.load_weights('my_mlp_weights.npz')
    network.testing(X_test, Y_test)
    # max that i get 93.39% 