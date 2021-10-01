import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd

from sklearn import datasets

# x = np.arange(100)
# xLastTen = x[90:]
#
# xUpdate = np.arange(0, 1000, 10)
#
# xDotProduct = x.dot(x)
#
# xAsteriskProduct = x * x
#
# xReshape = xUpdate.reshape((10,10))
#
# yNew = np.arange(1, 11)
#
# zNew = xReshape * yNew[:, np.newaxis]
#
# print(zNew)
#
# for i in range(10):
#     plt.plot(zNew[i])
#
# plt.show()

iris_data = datasets.load_iris()
iris_data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

#Q6(a)
print(len((iris_data)))

#Q6(b,c)
print(iris_data.describe())

#Q7






