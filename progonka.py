import pprint
import numpy as np


A = [
    [1, 7, 0, 0],
    [8, 7, 7, 0],
    [0, 4, 9, -6],
    [0, 0, -3, 1]
]

B = [
    -43,
    -71,
    -154,
    36
]


def nevyazka(A, B, X):
    r = np.dot(A, X) - B
    R = 0
    for i in range(len(r)):
        R += r[i] ** 2
    R = np.sqrt(R)
    return R


def progonka(matrix_a, matrix_b):
    vector_c, vector_d = np.zeros(4), np.zeros(4)
    vector_c[0] = - matrix_a[0][1] / matrix_a[0][0]
    vector_d[0] = matrix_b[0] / matrix_a[0][0]
    for i in range(1, 3):
        vector_c[i] = - matrix_a[i][i + 1] / (matrix_a[i][i - 1] * vector_c[i - 1] + matrix_a[i][i])
        vector_d[i] = (matrix_b[i] - matrix_a[i][i - 1] * vector_d[i - 1]) / (matrix_a[i][i - 1] * vector_c[i - 1] + matrix_a[i][i])

    vector_c[-1] = 0
    vector_d[-1] = (matrix_b[-1] - matrix_a[-1][-2] * vector_d[-2]) / (matrix_a[-1][-2] * vector_c[-2] + matrix_a[-1][-1])

    vector_x = np.zeros(4)
    vector_x[-1] = vector_d[-1]
    for i in range(3, 0, -1):
        vector_x[i - 1] = vector_c[i - 1] * vector_x[i] + vector_d[i - 1]
    pprint.pprint(vector_x, width=10)


progonka(A, B)
print("Невязка: r = ", nevyazka(A, B, [0, 0, 0, 0]))