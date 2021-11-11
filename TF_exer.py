import matplotlib.pyplot as plt
import numpy as np
import seaborn
import tensorflow as tf

x = np.arange(100, step=.1)
y = x + 20 * np.sin(x/10)

plt.scatter(x,y)
plt.show()


n_samples = 1000
batch_size = 100

x = np.reshape(x, (n_samples, 1))
y = np.reshape(y, (n_samples, 1))

x = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope("linear-regre"):
    w = tf.get_variable("weights", (1,1))
