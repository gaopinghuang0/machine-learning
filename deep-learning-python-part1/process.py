'''
Process ecommerce_data.csv
'''
import numpy as np
import pandas as pd

def get_data():
	df = pd.read_csv('ecommerce_data.csv')
	# df.head()
	data = df.as_matrix()

	X = data[:, :-1]  # X includes everything exclude Y
	Y = data[:, -1]   # Y is the last column

	# normalize numerical columns (columns 1 and 2)
	X[:, 1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
	X[:, 2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

	# create a new matrix X2 with the correct number of columns
	N, D = X.shape
	X2 = np.zeros((N, D+3))
	X2[:, 0:(D-1)] = X[:, 0:(D-1)]  # non-categorical

	# one-hot categorical columns
	# convert time_of_day column into 4 columns (4 periods)
	for n in xrange(N):
		t = int(X[n, D-1])
		X2[n, t+D-1] = 1

	# another method
	# Z = np.zeros((N, 4))
	# Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
	# # X2[:, -4:] = Z
	# assert(np.abs(X2[:, -4:] - Z).sum() < 10e-10)

	return X2, Y

def get_binary_data():
	# return only the data from the first 2 classes (i.e., 0/1 of user_action)
	X, Y = get_data()
	X2 = X[Y <= 1]
	Y2 = Y[Y <= 1]
	return X2, Y2