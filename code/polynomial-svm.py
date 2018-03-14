# File name: polynomial-svm.py
# Author: Dirk Brink
# Date created: 14/03/2018
# Date last modified: 14/03/2018
# Python version: 3.6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, validation_curve

# File path containing the dataset
input_file = '../data/spambase.data'

dataset = np.loadtxt(input_file, delimiter=",")

# Split the dataset matrix into input and output values
X = dataset[:,:-1]
y = dataset[:,-1]

# Split the dataset into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

classifier = SVC(kernel='poly', verbose=3)

# Run perceptron model on data and make predictions on test data
y_pred = classifier.fit(X_train, y_train).predict(X_test)
prediction_accuracy = accuracy_score(y_test, y_pred)


# cross_validation = cross_val_score(classifier, X_train, y_train, cv=5)
param_range = np.logspace(3, 4, 2)
#train_scores, test_scores = validation_curve(SVC(kernel='poly'), X_train, y_train, "degree", param_range)


train_scores_mean = 1
train_scores_max = 1
test_scores_mean = 1
test_scores_max = 1


plt.title("Validation Curve with Perceptron")
plt.xlabel("alpha")
plt.ylabel("Score")
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.legend(loc="best")

plt.figure(2)


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


print ("Best training accuracy = %f \nBest validation accuracy = %f \nTest accuracy = %f" % (train_scores_max, test_scores_max, prediction_accuracy))

plt.show()
