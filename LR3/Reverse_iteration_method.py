import numpy as np
from devtool import *


def reverse_iter(matrix_a, eps):
    matrix_x = np.ones(len(matrix_a))
    a = max(matrix_x, key=abs)
    a_p = a
    counter = 0
    while True:
        counter += 1
        matrix_x = np.linalg.inv(matrix_a) @ (matrix_x / a)
        a = max(matrix_x, key=abs)
        if abs(a - a_p) < eps:
            break
        a_p = a
    print("Minimum eigenvalue by abs: ", 1 / a)
    print("Count: ", counter)


reverse_iter(matrix_new, 0.0001)
