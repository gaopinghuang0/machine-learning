# machine-learning
Learning machine learning.

## Sources
1. [lazyprogrammer/machine_learning_examples](https://github.com/lazyprogrammer/machine_learning_examples/)
2. Deep-learning-python-part1: [data-science-deep-learning-in-python](https://www.udemy.com/data-science-deep-learning-in-python/)

## Notes:
Under `deep-learning-python-part1/`
1. For `backprop.py`

Learn how to rewrite slow for-loop into fast matrix form. Also, we can learn how to update W2, b2, W1, b1 using backpropagation.


2. For `backprop.py` and `ann_train.py`

Learn how to rewrite `softmax`, `forward`, `predict`, `classification_rate`, and `cross_entropy` into compact matrix form.


3. For `logistic_softmax_train.py` and `ann_train.py`

logistic_softmax only involves W and b. So the forward mode is `softmax(X.dot(W) + b)`, and the gradient decent only updates W and b.

ANN (one-hidden-layer) forward involves W1, b1, W2, b2, while backpropagation updates all of them. 