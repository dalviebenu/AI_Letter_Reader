from mlxtend.data import loadlocal_mnist
from keras.models import load_model
import matplotlib.pyplot as plt
from numpy import mean, std
import numpy as np
from keras.utils import to_categorical
from keras import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import KFold

OUTPUT = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
          5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
          10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
          15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
          20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
          25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
          30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
          35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
          40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
          45: 'r', 46: 't'}

alpha = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
         5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
         10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
         15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
         20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
         25: 'Z'}


def visualize(X, Y):
    print('Training set: X=%s Y=%s' % (X.shape, Y.shape))

    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    print(Y[:9])
    plt.show()


def scale_pixels(train, test):
    # We convert the pixel data from 0 - 255 to 0 - 1 (grayscale)
    train_float = train.astype('float32')
    test_float = test.astype('float32')
    train_float = train_float / 255.0
    test_float = test_float / 255.0
    return train_float, test_float


# plot diagnostic learning curves, blue shows training set and orange shows test set.
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()


# summarize overall model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()


def format_data(X, labels):
    X_reshaped = np.reshape(X, (X.shape[0], 28, 28))
    trainX = np.transpose(X_reshaped, (0, 2, 1))

    # convert dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    labels = labels.reshape(labels.shape[0], 1)
    # One hot encode the labels
    # visualize(trainX, labels)
    # labels = labels - 1  # for letters data set only
    labels = to_categorical(labels)

    return trainX, labels


def load_data():
    # load the letters emnist dataset from local files. Not used any more.
    X, trainY = loadlocal_mnist(
        images_path='./data/emnist-letters-train-images-idx3-ubyte/emnist-letters-train-images-idx3-ubyte',
        labels_path='./data/emnist-letters-train-labels-idx1-ubyte/emnist-letters-train-labels-idx1-ubyte'
    )
    testx, testY = loadlocal_mnist(
        images_path='./data/emnist-letters-test-images-idx3-ubyte/emnist-letters-test-images-idx3-ubyte',
        labels_path='./data/emnist-letters-test-labels-idx1-ubyte/emnist-letters-test-labels-idx1-ubyte'
    )
    trainX, trainY = format_data(X, trainY)
    testX, testY = format_data(testx, testY)

    print('data load complete')
    print('trainX: ', trainX.shape, 'trainY: ', trainY.shape, 'testX: ', testX.shape, 'testY: ', testY.shape)
    return trainX, trainY, testX, testY


def load_new_data():
    # load the emnist balanced dataset from local files
    X, trainY = loadlocal_mnist(
        images_path='./data/emnist-balanced-train-images-idx3-ubyte/emnist-balanced-train-images-idx3-ubyte',
        labels_path='./data/emnist-balanced-train-labels-idx1-ubyte/emnist-balanced-train-labels-idx1-ubyte'
    )
    testx, testY = loadlocal_mnist(
        images_path='./data/emnist-balanced-test-images-idx3-ubyte/emnist-balanced-test-images-idx3-ubyte',
        labels_path='./data/emnist-balanced-test-labels-idx1-ubyte/emnist-balanced-test-labels-idx1-ubyte'
    )
    trainX, trainY = format_data(X, trainY)
    testX, testY = format_data(testx, testY)

    print('data load complete')
    print('trainX: ', trainX.shape, 'trainY: ', trainY.shape, 'testX: ', testX.shape, 'testY: ', testY.shape)
    return trainX, trainY, testX, testY


def create_model():
    # A CNN to extract features, max pooling to down sample image, 1000 node layer to interpret features
    # and 47 nodes as output
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))  # input
    model.add(MaxPooling2D((2, 2)))
    # VGG blocks
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(47, activation='softmax'))  # Output layer
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
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))

        scores.append(acc)
        histories.append(history)
    return scores, histories


def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_new_data()
    # prepare pixel data
    trainX, testX = scale_pixels(trainX, testX)
    # evaluate model
    scores, histories = test_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)


def final_model():
    trainX, trainY, testX, testY = load_new_data()
    # prepare pixel data
    trainX, testX = scale_pixels(trainX, testX)
    # create, fit, and save model
    model = create_model()
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=32, verbose=1)
    model.save('final_model_letters.h5')


def predict_model(name):
    trainX, trainY, testX, testY = load_new_data()
    # prepare pixel data
    trainX, testX = scale_pixels(trainX, testX)
    # create, fit, and save model
    model = load_model(name)
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))


if __name__ == '__main__':
    final_model()
    predict_model('final_model_letters.h5')
    # run_test_harness()
    # load_new_data()
