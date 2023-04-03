import numpy as np
from devtool import *
from Levelie_method import get_root
from copy import deepcopy


def fadeev(matrix_a):
    matrix_a1 = deepcopy(matrix_a)
    vector_sp = np.trace(matrix_a)
    matrix_b = [matrix_a1 - vector_sp * np.eye(len(matrix_a))]
    for i in range(1, len(matrix_a)):
        matrix_a1 = matrix_a @ matrix_b[-1]
        p = np.trace(matrix_a1) / (i + 1)
        matrix_b.append(matrix_a1 - p * np.eye(len(matrix_a)))
    print("Default inversion matrix:", '\n', np.linalg.inv(matrix_a), '\n')
    print("Fadeev method inversion matrix:", '\n',
          (matrix_b[-2] / (np.trace(matrix_a1) / len(matrix_a))), '\n')
    print("Eigenvalues of the matrix:")
    for j in range(len(matrix_a)):
        vector_r = []
        root = get_root(matrix_a)
        vector_rk = root[j] ** (len(matrix_a) - 1) * np.eye(len(matrix_a))
        for i in range(len(matrix_a)):
            vector_rk += root[j] ** (len(matrix_a) - i - 2) * matrix_b[i]
        for i in range(len(matrix_a)):
            vector_r.append(vector_rk[i][0])
        print(f'R_{j + 1} =', vector_r)


fadeev(matrix_new)

