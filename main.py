import numpy
from mlxtend.data import loadlocal_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import KFold
from numpy import mean, std, indices
import numpy as np
from CNN import *
import scipy.io as sio


def format_data(X):
    X_reshaped = np.reshape(X, (X.shape[0], 28, 28))
    trainX = np.transpose(X_reshaped, (0, 2, 1))
    return trainX


def load_data():
    # load the emnist dataset from local files
    X, trainY = loadlocal_mnist(
        images_path='emnist-letters-train-images-idx3-ubyte',
        labels_path='emnist-letters-train-labels-idx1-ubyte'
    )
    testx, testY = loadlocal_mnist(
        images_path='emnist-letters-test-images-idx3-ubyte',
        labels_path='emnist-letters-test-labels-idx1-ubyte'
    )
    trainX = format_data(X)
    testX = format_data(testx)

    visualize(trainX, trainY)
    return trainX, trainY, testX, testY


load_data()
