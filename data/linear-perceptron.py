import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_file = 'spambase.data'

dataset = np.loadtxt(input_file, delimiter=",")

plt.figure(1, figsize=(8, 6))
plt.clf()

plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, -1], cmap=plt.cm.Set1, edgecolor='k')



print (dataset[:,-1])

plt.show()
