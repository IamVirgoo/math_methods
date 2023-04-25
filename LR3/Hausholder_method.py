import numpy as np
from copy import deepcopy
from devtool import *


def hausholder(matrix_a, eps):
    matrix_r = deepcopy(matrix_a)
    matrix_q = np.eye(len(matrix_a))
    matrix_b = deepcopy(matrix_a)
    for k in range(len(matrix_a) - 1):
        x = deepcopy(matrix_r)[:, k]
        for i in range(k):
            x[i] = 0
        e = np.zeros(len(matrix_a))
        e[k] = 1
        u = x - np.linalg.norm(x) * e
        matrix_p = np.eye(len(matrix_a)) - (np.dot((2 * u).reshape(len(matrix_a), 1), u.reshape(1, len(matrix_a)))) / np.linalg.norm(u) ** 2
        matrix_q = matrix_q @ matrix_p
        matrix_r = matrix_p @ matrix_r
        matrix_b = matrix_r @ matrix_q
    delta = 0
    for i in range(len(matrix_a)):
        delta += (matrix_b[i][i] - matrix_a[i][i]) ** 2
    delta = sqrt(delta)
    if delta >= eps:
        hausholder(matrix_b, eps)
    else:
        print("Matrix Eigenvalues:", '\n')
        for i in range(len(matrix_a)):
            print(matrix_b[i][i])
        print('\n', "Matrix A:", '\n', matrix_a, '\n')
        print("Matrix P:", '\n', matrix_p, '\n')
        print("Matrix Q:", '\n', matrix_q, '\n')
        print("Matrix R:", '\n', matrix_r, '\n')
        print("Matrix B:", '\n', matrix_b, '\n')


hausholder(matrix_new, 0.0001)
