#lab 1-2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd


xdata = 7 * np.random.random(100)
ydata = np.sin(xdata) + 0.25 * np.random.random(100)
zdata = np.exp(xdata) + 0.25 * np.random.random(100)

fig = plt.figure(figsize=(9, 6))

#create 3-D container
ax = plt.axes(projection='3d')
#visualize 3D scatter plot
ax.scatter3D(xdata, ydata, zdata)
#give labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.savefig('3d.png', dpi=300)



x = np.arange(100)
xLastTen = x[90:]

xUpdate = np.arange(0, 1000, 10)

xDotProduct = x.dot(x)

#element wise *
xAsteriskProduct = x * x
print('x*x')
print(xAsteriskProduct)

xReshape = xUpdate.reshape((10,10))
print('xReshape')
print(xReshape)

yNew = np.arange(1, 11)
print('yNew')
print(yNew)

zNew = xReshape * yNew[:, np.newaxis] #np.newaxis: yNew[1x10]-> yNew*[10x10]
print('zNew')
print(zNew)

for i in range(10):
    plt.plot(zNew[i])
plt.show()

for i in range(10):
    ax = plt.subplot(5, 2, i+1)
    plt.plot(zNew[i])
plt.show()
plt.savefig('lab2.png')

## W1 Tut Exercises
from sklearn import datasets

iris_data = datasets.load_iris()
iris_data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

#Q6(a)
print(len((iris_data)))

#Q6(b,c)
print(iris_data.describe())

#Q7






