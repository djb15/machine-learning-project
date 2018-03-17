# File name: mlp.py
# Author: Dirk Brink
# Date created: 17/03/2018
# Date last modified: 17/03/2018
# Python version: 3.6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV

# File path containing the dataset
input_file = '../data/spambase.data'

dataset = np.loadtxt(input_file, delimiter=",")

# Split the dataset matrix into input and output values
X = dataset[:,:-1]
# Preprocessing data is key for SVC
X_train_minmax = preprocessing.scale(X)

y = dataset[:,-1]

# Split the dataset into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X_train_minmax, y)

param_grid = [{"activation": ["relu", "identity", "logistic"], "alpha": [0.0001, 0.001, 0.01, 0.1, 1], "solver": ["lbfgs", "sgd", "adam"]}
            ]
clf = GridSearchCV(MLPClassifier(max_iter=10000), param_grid)
clf = clf.fit(X_train, y_train)
classifier = clf.best_estimator_
print (classifier)

# Run perceptron model on data and make predictions on test data
y_pred = classifier.fit(X_train, y_train).predict(X_test)
prediction_accuracy = accuracy_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
cnf_matrix = confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot the normalised confusion matrix
plot_confusion_matrix(cnf_matrix, classes=["Not Spam", "Spam"], normalize=True,
                      title='Normalized confusion matrix')


print ("Test accuracy = %f" % (prediction_accuracy))

plt.show()
