import copy

import numpy as np


A = [
    [0.11, 0.89, 0],
    [0.84, 12.57, 1.54],
    [1.27, 0.05, 8.07],
]

B = [
    0.06,
    3.15,
    1.89
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