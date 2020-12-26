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


def format_data(X, labels):
    X_reshaped = np.reshape(X, (X.shape[0], 28, 28))
    trainX = np.transpose(X_reshaped, (0, 2, 1))

    # convert dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    # One hot encode the labels
    labels = to_categorical(labels)
    return trainX, labels


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
    trainX, trainY = format_data(X, trainY)
    testX, testY = format_data(testx, testY)

    print('data load complete\n')
    print('trainX: ', trainX.shape, 'trainY: ', trainY.shape, 'testX: ', testX.shape, 'testY: ', testY.shape)
    return trainX, trainY, testX, testY


def create_model():
    # A CNN to extract features, max pooling to down sample image, 100 node layer to interpret features
    # and 10 nodes as output
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    # VGG block
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(27, activation='softmax'))
    # Gradient Descent learning model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def test_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # 5 fold cross validation is used, KFold.split returns indices. the indicies can be applied to labels
    # as they are ordered the same as the features
    kfold = KFold(n_folds, shuffle=True, random_state=1)

    for train_ix, test_ix in kfold.split(dataX):
        model = create_model()
        trainX, testX, trainY, testY = dataX[train_ix], dataX[test_ix], dataY[train_ix], dataY[test_ix]
        # history hold the performance at each epoch, will be added to histories to evaluate
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))

        scores.append(acc)
        histories.append(history)
    return scores, histories


def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_data()
    # prepare pixel data
    trainX, testX = scale_pixels(trainX, testX)
    # evaluate model
    scores, histories = test_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)


if __name__ == '__main__':
    run_test_harness()
