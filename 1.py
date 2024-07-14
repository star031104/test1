import numpy as np
import struct
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.utils import gen_batches
from sklearn.metrics import classification_report, confusion_matrix
from typing import *
from numpy.linalg import *

train_image_file = './data/MNIST/raw/train-images-idx3-ubyte'
train_label_file = './data/MNIST/raw/train-labels-idx1-ubyte'
test_image_file = './data/MNIST/raw/t10k-images-idx3-ubyte'
test_label_file = './data/MNIST/raw/t10k-labels-idx1-ubyte'


def decode_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(-1, 784)
        images = np.array(images, dtype=float)
    return images


def decode_label(path):
    with open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        labels = np.array(labels, dtype=float)
    return labels


def load_data():
    train_X = decode_image(train_image_file)
    train_Y = decode_label(train_label_file)
    test_X = decode_image(test_image_file)
    test_Y = decode_label(test_label_file)
    return train_X, train_Y, test_X, test_Y


trainX, trainY, testX, testY = load_data()

num_train, num_feature = trainX.shape
plt.figure(1, figsize=(20, 10))
for i in range(8):
    idx = np.random.choice(range(num_train))
    plt.subplot(int('24' + str(i + 1)))
    plt.imshow(trainX[idx, :].reshape((28, 28)))
    plt.title('label is %d' % trainY[idx])
plt.show()

trainX, testX = trainX / 255, testX / 255


def to_onehot(y):
    y = y.astype(int)
    num_class = len(set(y))
    Y = np.eye((num_class))
    return Y[y]


trainY = to_onehot(trainY)
testY = to_onehot(testY)
num_train, num_feature = trainX.shape
num_test, _ = testX.shape
_, num_class = trainY.shape
print('number of features is %d' % num_feature)
print('number of classes is %d' % num_class)
print('number of training samples is %d' % num_train)
print('number of testing samples is %d' % num_test)

from abc import ABC, abstractmethod, abstractproperty


class Activation(ABC):
    @abstractmethod
    def value(self, x: np.ndarray) -> np.ndarray:
        """
        Value of the activation function when input is x.
        Parameters:
          x is an input to the activation function.
        Returns:
          Value of the activation function. The shape of the return is the same as that of x.
        """
        return x

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the activation function with input x.
        Parameters:
          x is the input to activation function
        Returns:
          Derivative of the activation function w.r.t x.
        """
        return x


class Identity(Activation):
    """
    Identity activation function. Input and output are identical.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        n, m = x.shape
        return np.ones((n, m))


class Sigmoid(Activation):
    """
    Sigmoid activation function y = 1/(1 + e^(x*k)), where k is the parameter of the sigmoid function
    """

    def __init__(self, k: float = 1.):
        """
        Parameters:
          k is the parameter of the sigmoid function.
        """
        self.k = k
        super(Sigmoid, self).__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters:
          x is a two-dimensional numpy array.
        Returns:
          element-wise sigmoid value of the two-dimensional array.
        """
        val = 1 / (1 + np.exp(np.negative(x * self.k)))
        return val

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters:
          x is a two-dimensional array.
        Returns:
          a two-dimensional array whose shape is the same as that of x. The returned value is the elementwise
          derivative of the sigmoid function w.r.t. x.
        """
        val = 1 / (1 + np.exp(np.negative(x * self.k)))
        der = val * (1 - val)
        return der


class ReLU(Activation):
    """
    Rectified linear unit activation function
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        val = x * (x >= 0)
        return val

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        The derivative of the ReLU function w.r.t. x. Set the derivative to 0 at x=0.
        Parameters:
          x is the input to ReLU function
        Returns:
          elementwise derivative of ReLU. The shape of the returned value is the same as that of x.
        """
        der = np.ones(x.shape) * (x >= 0)
        return der


class Softmax(Activation):
    """
    softmax nonlinear function.
    """

    def __init__(self):
        super(Softmax, self).__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters:
          x is the input to the softmax function. x is a two-dimensional numpy array.
          Each row is the input to the softmax function
        Returns:
          output of the softmax function. The returned value is with the same shape as that of x.
        """
        n, k = x.shape
        beta = x.max(axis=1).reshape((n, 1))
        tmp = np.exp(x - beta)
        numer = np.sum(tmp, axis=1, keepdims=True)
        val = tmp / numer
        return val

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters:
          x is the input to the softmax function. x is a two-dimensional numpy array.
        Returns:
          a two-dimensional array representing the derivative of softmax function w.r.t. x.
        """
        n, k = x.shape
        D = np.zeros((k, k, n))
        for i in range(n):
            tmp = x[i:i + 1, :]
            val = self.value(x)
            D[:, :, i] = np.diag(val.reshape(-1)) - val.T.dot(val)
            return D.reshape((k, k, n)).transpose((2, 0, 1))


class Loss(ABC):
    """
    Abstract class for a loss function
    """

    @abstractmethod
    def value(self, yhat: np.ndarray, y: np.ndarray) -> float:
        """
        Value of the empirical loss function.
        Parameters:
          y_hat is the output of a neural network. The shape of y_hat is (n, k).
          y contains true labels with shape (n, k).
        Returns:
          value of the empirical loss function.
        """
        return 0

    @abstractmethod
    def derivative(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Derivative of the empirical loss function with respect to the predictions.
        Parameters:

        Returns:
          The derivative of the loss function w.r.t. y_hat. The returned value is a two-dimensional array with
          shape (n, k)
        """
        return yhat


class CrossEntropy(Loss):
    """
    Cross entropy loss function
    """

    def value(self, yhat: np.ndarray, y: np.ndarray) -> float:
        # m = y.shape[0]
        yhat = np.clip(yhat, 0.0001, 0.9999)  # 同逻辑回归的截断操作
        # 计算交叉熵损失的总和
        # los = -np.sum(y * np.log(yhat))

        # 对每个样本计算交叉熵损失，取这些损失值的平均值
        los = -np.mean(np.multiply(np.log(yhat), y) + np.multiply(np.log(1 - yhat), (1 - y)))
        return los

    def derivative(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        der = (yhat - y) / (yhat * (1 - yhat))
        return der


class CEwithLogit(Loss):
    """
    Cross entropy loss function with logits (input of softmax activation function) and true labels as inputs.
    """

    def value(self, logits: np.ndarray, y: np.ndarray) -> float:
        # 四种损失值计算方式
        # m = y.shape[0]
        # logits = np.clip(logits, 0.0001, 0.9999) #同逻辑回归的截断操作
        # los = (-1/m) * np.sum(y * logits）
        # los = np.sum(y * logits)- np.log(np.sum(np.exp(logits)))
        # delta = 1e-7
        # los = np.sum(-np.log(logits + delta) * y - np.log(1 - logits + delta) * (1 - y)) / m
        # los = -np.sum(y * np.log(logits) + (1 - y) * np.log(1 - logits)) / m
        n, k = y.shape
        beta = logits.max(axis=1).reshape((n, 1))
        tmp = logits - beta
        tmp = np.exp(tmp)
        tmp = np.sum(tmp, axis=1)
        tmp = np.log(tmp + 1.0e-40)
        los = -np.sum(y * logits) + np.sum(beta) + np.sum(tmp)
        los = los / n
        return los

    def derivative(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, k = y.shape
        beta = logits.max(axis=1).reshape((n, 1))
        tmp = logits - beta
        tmp = np.exp(tmp)
        numer = np.sum(tmp, axis=1, keepdims=True)
        yhat = tmp / numer
        der = (yhat - y) / n
        return der


def accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    Accuracy of predictions, given the true labels.
    Parameters:
      y_hat is a two-dimensional array. Each row is a softmax output.
      y is a two-dimensional array. Each row is a one-hot vector.
    Returns:
      accuracy which is a float number
    """
    n = y.shape[0]
    acc = np.sum(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)) / n
    return acc


# the following code implements a three layer neural network, namely input layer, hidden layer and output layer
digits = 10  # number of classes
_, n_x = trainX.shape
n_h = 16  # number of nodes in the hidden layer
learning_rate = 0.0001

sigmoid = ReLU()  # activation function in the hidden layer
softmax = Softmax()  # nonlinear function in the output layer
loss = CEwithLogit()  # loss function
epoches = 100

# initialization of W1, b1, W2, b2
W1 = np.random.randn(n_x, n_h)
b1 = np.random.randn(1, n_h)
W2 = np.random.randn(n_h, digits)
b2 = np.random.randn(1, digits)

# training procedure
for epoch in range(epoches):
    Z1 = np.dot(trainX, W1) + b1
    A1 = sigmoid.value(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax.value(Z2)
    cost = loss.value(Z2, trainY)

    dZ2 = loss.derivative(Z2, trainY)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid.derivative(Z1)
    dW1 = np.dot(trainX.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if (epoch % 100 == 0):
        print("Epoch", epoch, "cost: ", cost)

print("Final cost:", cost)

# testing procedure
Z1 = np.dot(testX, W1) + b1
A1 = sigmoid.value(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = softmax.value(Z2)

predictions = np.argmax(A2, axis=1)
labels = np.argmax(testY, axis=1)

# print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))


