
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Credit: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
def OneHotEncode1DLabel(y):
    # if y is categorical data
    # convert categories to numerical values first
    lbl_enc = LabelEncoder()
    lbl_enc.fit(y)
    n_classes = len(list(lbl_enc.classes_))
    
    return np.eye(n_classes)[y]

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z

def classification_rate(Y, P):
    return np.mean(Y == P)

def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()

def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]

    # # slow
    # ret1 = np.zeros((M, K))
    # for n in xrange(N):
    #     for m in xrange(M):
    #         for k in xrange(K):
    #             ret1[m,k] += (T[n,k] - Y[n,k])*Z[n,m]


    # note: optimize step-by-step from nested for-loops
    ret4 = Z.T.dot(T - Y)
    return ret4

def derivative_b2(T, Y):
    return (T-Y).sum(axis=0)

def derivative_w1(X,Z,T,Y,W2):
    N, D = X.shape
    M, K = W2.shape

    # slow
    # ret1 = np.zeros((D, M))
    # for n in xrange(N):
    #     for k in xrange(K):
    #         for m in xrange(M):
    #             for d in xrange(D):
    #                 ret1[d,m] += (T[n,k] - Y[n,k])*W2[m,k]*Z[n,m]*(1-Z[n,m])*X[n,d]
    # return ret1
    
    # fastest
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    ret2 = X.T.dot(dZ)

    # assert(np.abs(ret1 - ret2).sum() < 0.00001)

    return ret2

def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1-Z)).sum(axis=0)


def main():
    Nclass = 500

    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])
    print(X.shape)

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

    D = 2
    M = 4
    K = 3

    T = OneHotEncode1DLabel(Y)
    print(T)

    # plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    # plt.show()

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 10e-7
    costs = []
    for epoch in range(100000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print('cost:', c, 'classification_rate:', r)
            costs.append(c)

        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * derivative_b1(T, output, W2, hidden)

    plt.plot(costs)
    plt.show()



if __name__ == '__main__':
    main()

