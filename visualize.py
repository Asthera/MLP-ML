import matplotlib.pyplot as plt

def visualize(epochs, network):

    epochs = range(1, epochs+1)

    # create a subplot for loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, network.loss_values, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # create a subplot for accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, network.acc_values, 'r-')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.show()