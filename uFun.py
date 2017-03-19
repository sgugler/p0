import numpy as np
from numpy import clip, log, sqrt, square, mean, inf, std, linspace
from tempfile import mkdtemp
import os.path as path

# Generic File Extractor
# filepath ist das komplette prefix bis zur Zahl
# low ist die erste Zahl
# high ist die letzte Zahl

def get_data_4D(filepath, high):
    X = [];
    for i in range(0, high):
        lol = nib.load(filepath + str(i + 1) + ".nii")
        X.append(lol)
    return X
def get_train_data_X_4D():
    return get_data_4D("../data/set_train/train_", 278)
def get_train_data_X_1D():
    return [x.get_data().ravel() for x in get_train_data_X_4D()]
def get_train_data_Y():
    return np.genfromtxt('../data/targets.csv', delimiter='\n')
def get_test_data_X_4D():
    return get_data_4D("../data/set_test/test_", 138)
def get_test_data_X_1D():
    return [x.get_data().ravel() for x in get_test_data_X_4D()]
# our scoring function ( mean squared error )
def meanscore(y1, y2):
    return mean(square(y1 - y2))
def write_prediction(Ypred):
    prediction = ["ID,Prediction"]
    for i in range(0, 138):
        prediction.append(str(i + 1) + "," + str(Ypred[i]))
    np.savetxt('../data/prediction.csv', prediction, delimiter="\n", fmt="%s")