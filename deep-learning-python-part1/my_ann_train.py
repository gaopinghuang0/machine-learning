'''
train neural network on: ecommerce_data.csv
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from process import get_data

# Credit: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
def OneHotEncode1DLabel(y):
    # if y is categorical data
    # convert categories to numerical values first
    lbl_enc = LabelEncoder()
    lbl_enc.fit(y)
    n_classes = len(list(lbl_enc.classes_))
    
    return np.eye(n_classes)[y]

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))


X, Y = get_data()
Y = Y.astype(np.int32)


eval_size = 0.10
kf = KFold(len(Y), round(1. / eval_size))
train_indices, valid_indices = next(iter(kf))
Xtrain, Ytrain = X[train_indices], Y[train_indices]
Xtest, Ytest = X[valid_indices], Y[valid_indices]

Ytrain_ind = OneHotEncode1DLabel(Ytrain)

Ytest_ind = OneHotEncode1DLabel(Ytest)


M = 5
D = X.shape[1]
K = len(set(Y))

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)


# train loop
train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
    pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)

    ctrain = cross_entropy(Ytrain_ind, pYtrain)
    ctest = cross_entropy(Ytest_ind, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # gradient descent
    W2 -= learning_rate*Ztrain.T.dot(pYtrain - Ytrain_ind)
    b2 -= learning_rate*(pYtrain - Ytrain_ind).sum(axis=0)
    dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain*Ztrain)
    W1 -= learning_rate*Xtrain.T.dot(dZ)
    b1 -= learning_rate*dZ.sum(axis=0)
    if i % 1000 == 0:
        print(i, ctrain, ctest)

print("Final train classification_rate:", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, predict(pYtest)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()