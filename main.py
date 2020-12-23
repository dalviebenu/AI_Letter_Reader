from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import KFold
from numpy import mean, std


def visualize(X, Y):
    print('Training set: X=%s Y=%s' % (X.shape, Y.shape))

    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X[i], cmap=plt.get_cmap('gray'))

    plt.show()


def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # convert dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # One hot encode the labels
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def scale_pixels(train, test):
    # We convert the pixel data from 0 - 255 to 0 - 1 (grayscale)
    train_float = train.astype('float32')
    test_float = test.astype('float32')
    train_float = train_float / 255.0
    test_float = test_float / 255.0
    return train_float, test_float


def define_model():
    # A CNN to extract features, max pooling to down sample image, 100 node layer to interprit features
    # and 10 nodes as output
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # Gradient Descent learning model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # 5 fold cross validation is used, KFold.split returns indices. the indicies can be applied to labels
    # as they are ordered the same as the features
    kfold = KFold(n_folds, shuffle=True, random_state=1)

    for train_ix, test_ix in kfold.split(dataX):
        model = define_model()
        trainX, testX, trainY, testY = dataX[train_ix], dataX[test_ix], dataY[train_ix], dataY[test_ix]
        # history hold the performance at each epoch, will be added to histories to evaluate
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))

        scores.append(acc)
        histories.append(history)
    return scores, histories


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


# run the test for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = scale_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)


if __name__ == "__main__":
    run_test_harness()
