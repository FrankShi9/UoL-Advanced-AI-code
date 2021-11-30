# import itertools
# import numpy as np
#
# m = np.diag([0.5, 0.5, 0.5, 0.5])*4
#
# print(m)
#
# for i in range(1,5):
#     for ii in list(itertools.combinations([0, 1, 2, 3], i)):
#         print(ii)
#         for jj in ii:
#             print(m[jj])
# ipython:
import torch
#         a = torch.arange(9, dtype= torch.float) - 4
#         b = a.reshape((3, 3))
#         torch.norm(a)
#         #tensor(7.7460)
#         torch.norm(b)
#         #tensor(7.7460)
#         torch.norm(a, float('inf'))
#         #tensor(4.)

# x = torch.randn(4, 4)
# print(x)
# y = x.view(16)
# print(y)