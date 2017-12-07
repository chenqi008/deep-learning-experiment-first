from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import argparse

# Training settings
parser = argparse.ArgumentParser(description='experiments of deep learning')
parser.add_argument('--dataset', default='housing_scale', type=str, help='housing_scale|australian_scale')
parser.add_argument('--lr', type=float, default=0.1, help='Learning Rate. Default=0.001')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
opt = parser.parse_args()
print(opt)

def get_data(dataset):
    data = load_svmlight_file("../../dataset/{}".format(dataset))
    # data = load_svmlight_file("../dataset/australian_scale")
    input_data = data[0].toarray()
    return input_data, data[1]

def get_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    return x_train, x_test, y_train, y_test

def initial_w(dimension):
    w = np.zeros((dimension, 1))
    # w = np.random.random((dimension, 1))
    # w = np.random.normal(size=(dimension, 1))
    return w

def plot_loss(training_loss, testing_loss):
    x = np.array(range(1, len(training_loss)+1))
    plt.figure()
    plt.plot(x, np.array(training_loss), label="train")
    plt.plot(x, np.array(testing_loss), label="test")
    plt.xlabel("Epoch")
    plt.ylabel("loss of L2 norm")
    plt.title("Experiment")
    plt.legend()
    plt.show()


def main():
    # get dataset
    input_data, label = get_data(opt.dataset)
    x_train, x_test, y_train, y_test = get_dataset(input_data, label)

    # initialize the w
    w = initial_w(dimension=(13+1))

    # handle b
    train_column = np.ones((len(x_train), 1))
    x_train = np.column_stack((x_train, train_column))
    test_column = np.ones((len(x_test), 1))
    x_test = np.column_stack((x_test, test_column))

    # plot
    training_loss_list = []
    testing_loss_list = []

    for i in range(opt.nEpochs):
        # calculate the gradient
        grad = (np.dot(np.transpose(x_train), (np.dot(x_train, w) - y_train.reshape(-1, 1))))/len(x_train)
        # update
        w = w - opt.lr * grad
        # training error and testing error
        training_loss = (1.0/2)*(np.mean((np.square(np.dot(x_train, w) - y_train.reshape(-1, 1)))))
        testing_loss = (1.0/2)*(np.mean((np.square(np.dot(x_test, w) - y_test.reshape(-1, 1)))))
        training_loss_list.append(training_loss)
        testing_loss_list.append(testing_loss)
        print("training error:[{}] testing error:[{}]".format(training_loss, testing_loss))

    # plot
    plot_loss(training_loss_list, testing_loss_list)


if __name__ == '__main__':
    main()


