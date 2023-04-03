import numpy as np
from devtool import *
from copy import deepcopy


def danilevskiy(matrix_a):
    matrix_a1 = deepcopy(matrix_a)
    for i in range(len(matrix_a) - 1):
        matrix_m = np.eye(len(matrix_a))
        for j in range(len(matrix_a)):
            if j == len(matrix_a) - 2 - i:
                matrix_m[-2 - i][j] = 1 / matrix_a1[-1 - i][-2 - i]
            else:
                matrix_m[-2 - i][j] = - matrix_a1[-1 - i][j] / matrix_a1[-1 - i][-2 - i]
        matrix_m_inversion = np.linalg.inv(matrix_m)
        matrix_a1 = matrix_m_inversion @ matrix_a1 @ matrix_m
    print("Frobeniuss form:", '\n', matrix_a1)
    print("Eigenvalues of the matrix:", '\n', get_eigenvalues(matrix_a1[0]))


danilevskiy(matrix_new)