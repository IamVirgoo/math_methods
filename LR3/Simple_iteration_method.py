import numpy as np
from devtool import *


def simple_iteration(matrix_a, eps):
    matrix_x = [1] * len(matrix_a)
    matrix_y = matrix_a @ matrix_x
    matrix_l = matrix_y @ matrix_x
    matrix_x = matrix_y / np.linalg.norm(matrix_y)
    matrix_lp = matrix_l
    while True:
        matrix_y = matrix_a @ matrix_x
        matrix_l = matrix_y @ matrix_x
        matrix_x = matrix_y / np.linalg.norm(matrix_y)
        if abs(matrix_l - matrix_lp) < eps:
            break
        matrix_lp = matrix_l
    print("Maximum eigenvalue by abs: ", matrix_l)


simple_iteration(matrix_new, 0.0001)