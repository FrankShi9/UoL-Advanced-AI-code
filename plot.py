import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# f, ax = plt.subplots(1,1, figsize=(5,4))
#
# x = np.linspace(0,10,1000)
# y = np.power(x, 2)
# ax.plot(x,y)
# ax.set_xlim((1,5))
# ax.set_ylim((0,30))
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('$\Gamma$')
#
# plt.tight_layout()
# plt.savefig('line_plot_plus.pdf')

from mpl_toolkits.mplot3d import axes3d

ax = plt.subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.1)
ax.plot_wireframe(X, Y, Z, linewidth=0.1)

plt.savefig('wire.pdf')