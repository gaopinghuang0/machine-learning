# machine-learning
Learning machine learning.

## Sources
1. Code on Github: [lazyprogrammer/machine_learning_examples](https://github.com/lazyprogrammer/machine_learning_examples/)
2. Deep-learning-python-part1: [data-science-deep-learning-in-python](https://www.udemy.com/data-science-deep-learning-in-python/)
3. Facial-expression-recognition:
	1. Code on Github: [lazyprogrammer/facial-expression-recognition](https://github.com/lazyprogrammer/facial-expression-recognition)
	2. Dataset: [Kaggle facial expression](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)


## Notes:
Under `deep-learning-python-part1/`
1. For `backprop.py`

Learn how to rewrite slow for-loop into fast matrix form. Also, we can learn how to update W2, b2, W1, b1 using backpropagation.

2. For `backprop.py` and `ann_train.py`

Learn how to rewrite `softmax`, `forward`, `predict`, `classification_rate`, and `cross_entropy` into compact matrix form.

3. For `logistic_softmax_train.py` and `ann_train.py`

logistic_softmax only involves W and b. So the forward mode is `softmax(X.dot(W) + b)`, and the gradient decent only updates W and b.

ANN (one-hidden-layer) forward involves W1, b1, W2, b2, while backpropagation updates all of them. 
