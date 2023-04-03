import numpy as np
from devtool import *


def levelie(matrix_a, type):
    vector_k = [k + 1 for k in range(len(matrix_a))]
    vector_a_powers = [np.linalg.matrix_power(matrix_a, k + 1) for k in range(len(matrix_a))]
    vector_spk = [np.trace(vector_a_powers[i]) for i in range(len(matrix_a))]
    vector_pk = np.array((0, 0, 0, 0, 0))
    for i in range(len(matrix_a)):
        vector_pk[i] = vector_spk[i]
        for j in range(i):
            vector_pk[i] -= vector_spk[i - j - 1] * vector_pk[j]
        vector_pk[i] = vector_pk[i] / (i + 1)
    if type == "print":
        print('k: ', vector_k, '\n')
        print('a^k: ', vector_a_powers[4], '\n')
        print('Sk = Spk: ', vector_spk, '\n')
        print('pk: ', vector_pk, '\n')
        print('Eigenvalues: ', [round(N(solve)) for solve in get_eigenvalues(vector_pk)])
    return [round(N(solve)) for solve in get_eigenvalues(vector_pk)]


def get_root(matrix_):
    return levelie(matrix_, '')


levelie(matrix_new, 'print')
