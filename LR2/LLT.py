import numpy as np

A = np.array([
    [-1, 2, -1, 3],
    [2, 5, 0, 9],
    [-1, 0, 14, 15],
    [3, 9, 15, 59]
])

B = np.array([
    -3,
    -8,
    14,
    130
])


def nevyazka(A, B, X):
    r = np.dot(A, X) - B
    R = 0
    for i in range(len(r)):
        R += r[i] ** 2
    R = np.sqrt(R)
    return R


def decompose_llt(matrix_a, matrix_b):
    matrix_l = np.matrix(np.zeros([matrix_a.shape[0], matrix_a.shape[1]]))
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                matrix_l[0, 0] = np.sqrt(abs(matrix_a[i, j]))
            elif i == j:
                sum_ = 0
                for k in range(i):
                    sum_ += matrix_l[i, k] ** 2
                matrix_l[i, j] = np.sqrt(abs(matrix_a[i, j] - sum_))
            elif j == 0:
                matrix_l[i, j] = matrix_a[i, j] / matrix_l[j, j]
            else:
                if j <= i:
                    sum_ = 0
                    for k in range(i):
                        sum_ += matrix_l[i, k] * matrix_l[j, k]
                    matrix_l[i, j] = (matrix_a[i, j] - sum_) / matrix_l[j, j]
    lt = np.transpose(matrix_l)
    vector_y, vector_x = np.zeros(4), np.zeros(4)
    vector_y[0] = matrix_b[0] / matrix_l[0, 0]
    for i in range(4):
        sum_ = 0
        for k in range(i):
            sum_ += matrix_l[i, k] * vector_y[k]
        vector_y[i] = (matrix_b[i] - sum_) / matrix_l[i, i]
    vector_x[-1] = vector_y[-1]
    for i in range(3, -1, -1):
        sum_ = 0
        for k in range(i + 1, 4):
            sum_ += matrix_l[k, i] * vector_x[k]
        vector_x[i] = (vector_y[i] - sum_) / matrix_l[i, i]
    return vector_x


print(decompose_llt(A, B))
print("Невязка: r = ", nevyazka(A, B, [0, 0, 0, 0]))