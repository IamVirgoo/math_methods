import numpy as np
from devtool import *


def direct_iter(matrix_a, eps):
    matrix_x = np.ones(len(matrix_a))
    a = max(matrix_x, key=abs)
    a_p = a
    while True:
        matrix_x = matrix_a @ (matrix_x / a)
        a = max(matrix_x, key=abs)
        if abs(a - a_p) < eps:
            break
        a_p = a
    print("Maximum eigenvalue by abs: ", a)


print(direct_iter(matrix_new, 0.0001))
