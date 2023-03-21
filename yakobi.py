import copy

import numpy as np


A = [
    [2.09, 1.68, 6.17],
    [6.01, 0.11, 2.48],
    [2.27, 12.13, 2.27],
]

B = [
    1.75,
    0.86,
    1.45
]


def nevyazka(A, B, X):
    r = np.dot(A, X) - B
    R = 0
    for i in range(len(r)):
        R += r[i] ** 2
    R = np.sqrt(R)
    return R


def yacobi(A, B, eps):
    n = len(A)
    D = np.eye(n)
    F = copy.deepcopy(A)
    for i in range(n):
        for j in range(n):
            if i == j:
                F[i][j] = 0
                D[i][j] = A[i][j]
    D_inv = np.linalg.inv(D)
    C = np.dot(-D_inv, F)
    d = np.dot(D_inv, B)
    X = [0] * n
    r = nevyazka(A, B, X)
    while r > eps:
        X = np.dot(C, X) + d
        r = nevyazka(A, B, X)
    print("Метод Якоби.")
    print("X:", X)
    print("epsilon =", eps)
    print("Невязка: r = ", nevyazka(A, B, X))


yacobi(A, B, 0.001)