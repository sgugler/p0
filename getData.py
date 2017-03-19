import numpy as np


def loadTrain():
    f = open("train.csv")
    f.readline()
    data = np.loadtxt(f, delimiter=",")

    X = data[:, 2:]
    y = data[:, 1]

    return (X, y)


def loadTest():
    f = open("test.csv")
    f.readline()
    data = np.loadtxt(f, delimiter=",")

    X_test = data[:, 1:]
    # y = data[:, 0]

    return X_test
