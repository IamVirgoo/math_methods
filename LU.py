import numpy as np

A = np.array([
    [1, 5, 5, 3],
    [5, 50, 50, 40],
    [5, 50, 51, 45],
    [3, 40, 45, 63]
])

B = np.array([
    -76,
    -755,
    -773,
    -701
])


def nevyazka(A, B, X):
    r = np.dot(A, X) - B
    R = 0
    for i in range(len(r)):
        R += r[i] ** 2
    R = np.sqrt(R)
    return R


def decompose_lu(matrix_a):
    matrix_lu = np.matrix(np.zeros([matrix_a.shape[0], matrix_a.shape[1]]))
    for k in range(4):
        for j in range(k, 4):
            matrix_lu[k, j] = matrix_a[k, j] - matrix_lu[k, :k] * matrix_lu[:k, j]
        for i in range(k + 1, 4):
            matrix_lu[i, k] = (matrix_a[i, k] - matrix_lu[i, : k] * matrix_lu[: k, k]) / matrix_lu[k, k]
    return matrix_lu


def get_l(m):
    matrix_l = m.copy()
    for i in range(matrix_l.shape[0]):
        matrix_l[i, i] = 1
        matrix_l[i, i + 1:] = 0
    return np.matrix(matrix_l)


def get_u(m):
    matrix_u = m.copy()
    for i in range(1, matrix_u.shape[0]):
        matrix_u[i, :i] = 0
    return matrix_u


def solve_lu(lu_matrix, b, type):
    # get supporting vector y
    y = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(y.shape[0]):
        y[i, 0] = b[i] - lu_matrix[i, :i] * y[:i]
    # get vector of answers x
    x = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(1, x.shape[0] + 1):
        x[-i, 0] = (y[-i] - lu_matrix[-i, -i:] * x[-i:, 0]) / lu_matrix[-i, -i]

    return x if type == 'x' else y


LU = decompose_lu(A)
L = get_l(LU)
U = get_u(LU)
x = solve_lu(LU, B, 'x')
print(x)
print("Невязка: r = ", nevyazka(A, B, [0, 0, 0, 0]))