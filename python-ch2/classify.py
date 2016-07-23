# classify.py
'''
First classification model
'''

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# we load the data with load_iris from sklearn
data = load_iris()
print data
# features = data['data']
# feature_names = data['feature_names']
# target = data['target']

# # Visulization
# for t,marker,c in zip(xrange(3), '>ox', 'rgb'):
# 	plt.scatter(features[target == t, 0],
# 		features[target == t, 1],
# 		marker=marker,
# 		c=c)
