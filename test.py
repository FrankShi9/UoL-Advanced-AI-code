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