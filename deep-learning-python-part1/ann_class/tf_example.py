
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def OneHotEncode1DLabel(y, n_classes):
    # # if y is categorical data
    # # convert categories to numerical values first
    # lbl_enc = LabelEncoder()
    # lbl_enc.fit(y)
    # n_classes = len(list(lbl_enc.classes_))
    return np.eye(n_classes)[y]


Nclass = 500

D = 2
M = 4
K = 3

X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3]).astype(np.float32)

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

T = OneHotEncode1DLabel(Y, K)

# tensor flow variables are not the same as regular Python variables
def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))


def forward(X, W1, b1, W2, b2):
	Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
	return tf.matmul(Z, W2) + b2

tfX = tf.placeholder(tf.float32, [None, D])
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weights([D, M])
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights([K])

logits = forward(tfX, W1, b1, W2, b2)

cost = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(
		labels=tfY,
		logits=logits
		)
	)

# WARNING: This op expects unscaled logits,
# since it performs a softmax on logits
# internally for efficiency.
# Do not call this op with the output of softmax,
# as it will produce incorrect results.
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
# input parameter is the learning rate


predict_op = tf.argmax(logits, 1)
# input parameter is the axis on which to choose the max


# just stuff that has to be done
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
	sess.run(train_op, feed_dict={tfX: X, tfY: T})
	pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
	if i % 10 == 0:
		print(np.mean(Y == pred))















