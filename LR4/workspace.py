from decimal import Decimal, getcontext, ROUND_HALF_UP
import numpy as np
import math
import matplotlib.pyplot as plt
import sympy
from IPython import display
from sys import exit
import sympy as sp
import time
from scipy.optimize import minimize


# 21 вариант 3 и 5

def equation_for_chart(x):
    ret1 = 1.5 - 0.4 * x ** 1 / 3
    ret2 = -0.5 * np.log(x)
    return ret1, ret2


def equation(x):
    ret = 1.5 - 0.4 * x ** 1 / 3 - 0.5 * math.log(x)
    return ret


def yravnenie_proizvodnya(x):
    # x = sp.Symbol('x')
    # f = 0.5 * sp.exp(-1 * x ** 2) + x * sp.cos(x)
    # f_prime = sp.diff(f, x)
    # ret = f_prime.subs(x, x0)
    ret = -2 / (15 * x ** 2 / 3) - 1 / 2 * x
    return ret


def yravnenie_for_simple_iteration(x):
    ret = 2 * x ** 2 - x ** 3 - math.e ** x
    return ret


def yravnenie_for_simple_iteration_proizvodnya(x):
    ret = -3 / 5 * x ** 1 / 2 - 1 / 2 * x
    return ret


def yravnenie_proizvodnya2(x):
    ret = 4 * x - 3 * x ** 2 - math.e ** x
    return ret


# отрезок [-1;0] EPS = 0.000001


def graphic(a, b):
    # Создаем массив точек для построения графика
    x = np.linspace(a, b, 1000)
    y1, y2 = equation_for_chart(x)
    plt.plot(x, y1, label='y1 = 0.5 * pow(math.e, pow(-1 * x, 2))')
    plt.plot(x, y2, label='x * np.cos(x)')
    plt.legend()
    plt.title('Графики функций y1 и y2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([-30, 30, -10, 10])
    plt.show()


# Метод дихотомии
def method_dexotomiya(a, b, eps, lenght):
    while (lenght > eps):
        c = (a + b) / 2
        ax = equation(a)
        bx = equation(b)
        cx = equation(c)
        if (cx * ax > 0):
            a = c
        else:
            b = c
        lenght = b - a
        x = (a + b) / 2
    print("x*= ", x, "  f(x*)= ", equation(x), "   lengt= ", lenght)


# Метод Ньютона (касательных)
def method_Nuytona(a, b):
    xn = b
    xn1 = a
    while (abs(xn - xn1) > eps):
        xn = xn1
        xn1 = - equation(xn) / yravnenie_proizvodnya(xn) + xn
    print("xn1=", xn1, "    F(xn1)=", yravnenie_proizvodnya(xn1), "  xn - xn1=e", abs(xn - xn1))


# Метод простой итерации
def linezation(x, x0):
    return x - equation(x0) / yravnenie_proizvodnya(x0)


def simple_iteration_method(interval1, interval2, eps=0.000001):
    q = max([abs(yravnenie_proizvodnya(x)) for x in np.arange(interval1, interval2, eps)])
    if q >= 1:
        print('The function doesn\'t meet the second condition of convergence.')
        maximum = 7.3678

        def pr3(x):
            return -math.e ** x - 6 * x + 4

        Q = maximum
        k = math.ceil(Q / 2) * -1
        iters = 0
        print("q:", Q, "k:", k)

        def fq(x):
            return (x - (yravnenie_for_simple_iteration(x) / k))

        interval1 = -1
        interval2 = 0
        x_prev = interval1
        x_cur = fq(x_prev)
        print(x_prev, x_cur)
        while abs(x_cur - x_prev) >= eps:
            # print(fq(x_prev))
            print(x_cur)
            x_prev = x_cur
            x_cur = fq(x_prev)
            iters += 1
        return x_cur, iters
    ###
    x_prev = (interval1 + interval2) / 2
    x_cur = equation(x_prev)

    iters = 0
    while q / (1 - q) * abs(x_cur - x_prev) >= eps:
        x_prev = x_cur
        x_cur = equation(x_prev)
        iters += 1

    return x_cur, iters


# разностный метод Ньютона с постоянным шагом h = 0,001
def Newton_method(a, b):
    h = 0.001
    xn = b
    xn1 = a
    while (abs(xn - xn1) > eps):
        xn = xn1
        xn1 = xn - ((h * equation(xn)) / (equation(xn + h) - equation(xn)))
    print("xn1=", xn1, "    F(xn1)=", equation(xn1), "  xn - xn1=e", abs(xn - xn1))


# Вариант 21
eps = 0.000001

# 1.0
a = -100
b = 100
# graphic(a, b)
# 1.1
a, b = 4, 6
lenght = b - a
# print("Метод дихотомии")
# method_dexotomiya(a, b, eps, lenght)
# 1.2
# print("Метод Ньютона (Касательных)")
# method_Nuytona(a, b)
# 1.3
r, i = simple_iteration_method(a, b, eps)
print("itteranions =", i, " x = ", r)


# 1.4
# print("Метод Ньютона с постоянным шагом")
# Newton_method(a, b)

# Системы
def f1(x1, x2):
    return np.sin(x1 + 1.5) - x2 + 2.9


def f2(x1, x2):
    return np.cos(x2 - 2) + x1


a1, b1 = -100, 100
a2, b2 = -100, 100


def show_Systems(a1, b1, a2, b2):
    x = np.linspace(a, b, 1000)
    y1, y2 = equation_for_chart(x)
    plt.plot(x, y1, label='y1 = 0.5 * pow(math.e, pow(-1 * x, 2))')
    plt.plot(x, y2, label='x * np.cos(x)')
    plt.axis([-30, 30, -20, 20])

    x1 = np.linspace(a1, b1, 100)
    x2 = np.linspace(a2, b2, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # вычисляем значения функций F1 и F2 на сетке
    Z1 = f1(X1, X2)
    Z2 = f2(X1, X2)

    # строим график функций на прямоугольнике [[a1, b1]; [a2, b2]]
    plt.contour(X1, X2, Z1, levels=[0], colors='r')
    plt.contour(X1, X2, Z2, levels=[0], colors='b')

    # добавляем подписи осей и заголовок
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('График системы уравнений F1 и F2')

    # выводим график на экран
    plt.show()


###
def convergence_check_f(Jacobi_matrix, x):
    Jacobi_matrix_x = [[df_i(*x) for df_i in row] for row in Jacobi_matrix]
    if np.linalg.det(Jacobi_matrix_x) != 0:
        return True
    return False


def Newton_method_for_systems(f_array: tuple, Jacobi_matrix: list, start_point: list, eps: float) -> list:
    x_cur = np.array(start_point)
    iters = 0
    while True:
        if not convergence_check_f(Jacobi_matrix, x_cur):
            print('The condition of convergence hasn\'t been satistied: Jacobi matrix is not invertible.')
            return 0, 0

        f_array_cur = np.array([f_i(*x_cur.tolist()) for f_i in f_array])
        Jacobi_matrix_cur = [[df_i(*x_cur.tolist()) for df_i in row] for row in Jacobi_matrix]

        # J(x^k) * delta(x^k) = -f(x^k), solve with respect to delta(x^k)
        delta_x_k = np.linalg.solve(Jacobi_matrix_cur, -f_array_cur)

        # delta(x^k) = x^(x+1) - x^k, so we get x^(k+1) from this equation
        x_next = delta_x_k + x_cur
        iters += 1
        if np.linalg.norm(x_next - x_cur) < eps:
            return x_next.tolist(), iters
        x_cur = np.copy(x_next)


###

def convergence_check_f(Jacobi_matrix, x):
    Jacobi_matrix_x = [[df_i(*x) for df_i in row] for row in Jacobi_matrix]
    if np.linalg.det(Jacobi_matrix_x) != 0:
        return True
    return False


def modified_Newton_method_for_systems(f_array: tuple, Jacobi_matrix: list, start_point: list, eps: float) -> list:
    x_cur = np.array(start_point)
    iters = 0
    Jacobi_matrix_cur = []
    for row in Jacobi_matrix:
        cur_row = []
        for df_i in row:
            cur_row.append(df_i(*x_cur.tolist()))
        Jacobi_matrix_cur.append(cur_row)
    while True:
        if not convergence_check_f(Jacobi_matrix, x_cur):
            print('The condition of convergence hasn\'t been satistied: Jacobi matrix is not invertible.')
            return 0, 0
        f_array_cur = np.array([f_i(*x_cur.tolist()) for f_i in f_array])
        # J(x^k) * delta(x^k) = -f(x^k), solve with respect to delta(x^k)
        delta_x_k = np.linalg.solve(Jacobi_matrix_cur, -f_array_cur)

        # delta(x^k) = x^(x+1) - x^k, so we get x^(k+1) from this equation
        x_next = delta_x_k + x_cur
        iters += 1
        if np.linalg.norm(x_next - x_cur) < eps:
            return x_next.tolist(), iters
        x_cur = np.copy(x_next)


###
def convergence_check_phi(Jacobi_matrix: list, x: tuple) -> bool:
    Jacobi_matrix_x = [[phi(*x) for phi in row] for row in Jacobi_matrix]
    max_norm = max([np.linalg.norm(col) for col in np.array(Jacobi_matrix_x).T])
    if max_norm < 1:
        return True
    return False


def fi1(x2):
    return -math.cos(x2 - 2)


def fi2(x1):
    return math.sin(x1 + 1.5) + 2.9


def simple_iteration_method_for_systems(phi_array: tuple, Jacobi_matrix: list, start_point: list, eps: float) -> list:
    ''' Find a root point of the system of non-linear equations using the simple iteration method.
        Returns an approximate root and the number of iterations.'''
    if not convergence_check_phi(Jacobi_matrix, start_point):
        print('The condition of convergence hasn\'t been satistied.')
        x1 = (a1 + b1) / 2
        x2 = (a2 + b2) / 2
        iters = 0
        g1 = fi1(x2);
        g2 = fi2(x1);
        dx1 = x1 - g1;
        dx2 = x2 - g2;
        x1 = g1;
        x2 = g2;

        while (abs(dx1) > eps or abs(dx2) > eps):
            g1 = fi1(x2);
            g2 = fi2(x1);
            dx1 = x1 - g1;
            dx2 = x2 - g2;
            x1 = g1;
            x2 = g2;
            iters += 1
        return (x1, x2), iters
        #####

    x_prev = start_point.copy()
    x_cur = [phi_array[i](*x_prev) for i in range(len(x_prev))]

    iters = 0
    while np.linalg.norm(np.array(x_prev) - np.array(x_cur)) >= eps:
        if not convergence_check_phi(Jacobi_matrix, x_cur):
            print('The condition of convergence hasn\'t been satistied.')

        x_prev = x_cur.copy()
        x_cur = []
        for i in range(len(x_prev)):
            func_result = phi_array[i](*x_prev)
            x_cur.append(func_result)
        iters += 1

    return x_cur, iters


###
def newton_system_minimization(F, J, start_point, eps):
    x_cur = np.array(start_point)
    iters = 0
    while True:
        F_cur = np.array(F(*x_cur.tolist()))
        J_cur = np.array(J(*x_cur.tolist()))

        # H(x_n) * delta_x_n = -J(x_n)^T * F(x_n), solve with respect to delta_x_n
        H_cur = np.dot(np.transpose(J_cur), J_cur)
        delta_x_n = np.linalg.solve(H_cur, -np.dot(np.transpose(J_cur), F_cur))

        # x_{n+1} = x_n + delta_x_n
        x_next = delta_x_n + x_cur
        iters += 1

        if np.linalg.norm(F(*x_next.tolist())) < eps:
            return x_next.tolist(), iters
        x_cur = np.copy(x_next)


###

###
show_Systems(a1, b1, a2, b2)
###
# def newton_system
start_point = [0, 3]
f_array = (lambda x1, x2: np.sin(x1 + 1.5) - x2 + 2.9, lambda x1, x2: x1 + np.cos(x2 - 2))
Jacobi_matrix = [
    [lambda x1, x2: np.cos(x1 + 1.5), lambda x1, x2: -1],
    [lambda x1, x2: 1, lambda x1, x2: -np.sin(x2 - 2)]
]
root, iter_count = Newton_method_for_systems(f_array, Jacobi_matrix, start_point, eps)
print(f'Метод Ньютона: Root: {root}, iterations: {iter_count}')
###

root, iter_count = modified_Newton_method_for_systems(f_array, Jacobi_matrix, start_point, eps)
print(f'Модифицированный метод Ньютона: Root: {root}, iterations: {iter_count}')
###
start_point = [1, 3]
phi_array = (lambda x1, x2: np.sin(x1 + 1.5) - x2 + 2.9, lambda x1, x2: x1 + np.cos(x2 - 2))
Jacobi_matrix = [
    [lambda x1, x2: np.cos(x1 + 1.5), lambda x1, x2: -1],
    [lambda x1, x2: 1, lambda x1, x2: -np.sin(x2 - 2)]
]
eps = 0.000001
root, iter_count = simple_iteration_method_for_systems(phi_array, Jacobi_matrix, start_point, eps)
print(f'Простой метод Итерации Root: {root}, iterations: {iter_count}')


###
def F(x1, x2):
    return [np.sin(x1 + 1.5) - x2 + 2.9, x1 + np.cos(x2 - 2)]


# определяем ее якобиан
def J(x1, x2):
    return [[np.cos(x1 + 1.5), -1], [1, -np.sin(x2 - 2)]]


# задаем начальную точку
start_point = [1, 3]

# вызываем функцию метода Ньютона для решения системы нелинейных уравнений
result, iters = newton_system_minimization(F, J, start_point, eps)
print(f'методом Ньютона для нахождения минимума функции: Root: {result}, iterations: {iters}')
