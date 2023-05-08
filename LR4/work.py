import plotly.graph_objs as go
from math import cos, exp, sqrt
from sympy import *
import numpy as np
import matplotlib.pyplot as plt


def function(x):
    return (cos(x) ** 2) - 0.1 * exp(x)


def function_(x):
    return sqrt(x) - (x ** -1) * log(x) + 4 - 1.5


def system_function_first(x1, x2):
    return sin(x2 + 1) - x1 - 1


def system_function_second(x1, x2):
    return 2 * x2 + cos(x1) - 0.5


def find_bound(funct):
    left_bound = 0
    right_bound = 0
    while funct(left_bound) * funct(right_bound) > 0:
        right_bound += 1
    while funct(left_bound) * funct(right_bound) < 0:
        left_bound += 1
    left_bound -= 1
    while funct(left_bound) * funct(right_bound) < 0:
        right_bound -= 1
    right_bound += 1

    return left_bound, right_bound


def dihotomy(funct, left_bound=None, right_bound=None, eps=0.00001):
    if left_bound is None or right_bound is None:
        left_bound, right_bound = find_bound(funct)
    if right_bound - left_bound < eps:
        print("Метод Дихотомии: ", left_bound)
        return
    center = (left_bound + right_bound) / 2
    if funct(center) * funct(right_bound) < 0:
        left_bound = center
    else:
        right_bound = center
    dihotomy(funct, left_bound, right_bound, eps)


dihotomy(function, eps=0.00001)


def nyuton(funct, x=None, eps=0.00001):
    if x is None:
        x = find_bound(funct)[0]
    x_p = x
    argument = Symbol("x")
    while True:
        foo = funct(argument)
        x = N(- funct(x) / diff(foo, argument).subs(argument, x) + x_p)
        if abs(x - x_p) < eps:
            break
        x_p = x
    return x


print("Метод Ньютона: ", nyuton(function))


def simple_iteration(funct, left_bound=None, right_bound=None, eps=0.00001):
    if left_bound is None or right_bound is None:
        left_bound, right_bound = find_bound(funct)
    arg = Symbol('x')
    f = funct(arg)
    max_arg = left_bound
    if funct(left_bound) < funct(right_bound):
        max_value = funct(right_bound)
        max_arg = right_bound
    dif = [diff(f, arg).subs(arg, i) for i in np.arange(left_bound, right_bound, 0.001)]
    k = max(dif, key=abs) / 2
    '''print("Q: ", max(dif, key=abs))
    print(k)'''
    phi = arg - funct(arg) / k
    x_p = N(phi.subs(arg, right_bound))
    x = N(phi.subs(arg, x_p))
    while abs(x - x_p) > eps:
        x_p = x
        x = N(phi.subs(arg, x_p))
    return N(x)


print("Метод простой итерации: ", simple_iteration(function))


def hord(funct, x0=None, x1=None, eps=0.00001):
    if x0 is None or x1 is None:
        x0, x1 = find_bound(funct)
        x2p = x0
        x2 = x1
    while abs(x2p - x2) > eps:
        x2p = x2
        x2 = N(x0 - funct(x0) / (funct(x1) - funct(x0)) * (x1 - x0))
        if funct(x2) * funct(x0) < 0:
            x1 = x2
        else:
            x0 = x2
    return x2


print("Метод хорд: ", hord(function))


def nyuton_for_system(funcs, eps=0.000001):
    xs = [Symbol(f'x{i}') for i in range(len(funcs))]
    j = [[diff(func, X) for X in xs] for func in funcs]

    x = [1] * len(funcs)
    x_p = [1] * len(funcs)
    for _ in range(1000):
        j_x = np.array([[float(el.subs(xs[0], x_p[0]).subs(xs[1], x_p[1])) for el in line] for line in j])
        f_x = np.array([float(funcs[i].subs(xs[0], x_p[0]).subs(xs[1], x_p[1])) for i in range(len(funcs))])
        x = np.array(x) - np.linalg.inv(j_x) @ f_x
        if sqrt((x[0] - x_p[0]) ** 2 + (x[1] - x_p[1]) ** 2) < eps:
            return x
        x_p = x.copy()
    return None


x1, x2 = Symbol('x0'), Symbol('x1')

function_system_first = system_function_first(x1, x2)
function_system_second = system_function_second(x1, x2)

print("Метод Ньютона для систем уравнений: ", nyuton_for_system((function_system_first, function_system_second)))


def nyuton_double_step(funcs, eps=0.000001):
    xs = [Symbol(f'x{i}') for i in range(len(funcs))]
    j = [[diff(func, X) for X in xs] for func in funcs]

    x = [1] * len(funcs)
    z = [1] * len(funcs)
    for _ in range(1000):
        j_x = np.array([[float(el.subs(xs[0], x[0]).subs(xs[1], x[1])) for el in line] for line in j])
        f_x = np.array([float(funcs[i].subs(xs[0], x[0]).subs(xs[1], x[1])) for i in range(len(funcs))])
        z = np.array(x) - np.linalg.inv(j_x) @ f_x
        f_x = np.array([float(funcs[i].subs(xs[0], z[0]).subs(xs[1], z[1])) for i in range(len(funcs))])
        x = np.array(z) - np.linalg.inv(j_x) @ f_x
        if sqrt((x[0] - z[0]) ** 2 + (x[1] - z[1]) ** 2) < eps:
            return x
        x_p = x.copy()
    return None


print("Двуступенчатый метод Ньютона: ", nyuton_double_step((function_system_first, function_system_second)))

Fs1_x2 = solve(function_system_first, x2)[0]
Fs2_x2 = solve(function_system_second, x2)[0]


def graph():
    xs = list(np.arange(-10, 10, 0.1))

    ys1 = [complex(Fs1_x2.subs(x1, x)).real for x in xs]
    ys2 = [complex(Fs2_x2.subs(x1, x)).real for x in xs]

    plt.plot(xs, ys1, label='')
    plt.plot(xs, ys2, label='')

    plt.legend()
    plt.title('Graphics of y1 and y2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


xs = list(np.arange(-10, 10, 0.1))

ys1 = [complex(Fs1_x2.subs(x1, x)).real for x in xs]
ys2 = [complex(Fs2_x2.subs(x1, x)).real for x in xs]

xS0, xS1 = Symbol('x0'), Symbol('x1')

fig = go.Figure()

F1 = sin(xS1 + 1) - xS0 - 1
F2 = 2 * xS1 + cos(xS0) - 0.5

F1_x2 = solve(F1, xS1)
# print(F1_x2)
F2_x2 = solve(F2, xS1)
# print(F2_x2)

for f1 in F1_x2:
    for f2 in F2_x2:
        xs = list(np.arange(-10, 10, 0.1))

        ys1 = [complex(f1.subs(xS0, x)) for x in xs]
        ys2 = [complex(f2.subs(xS0, x)) for x in xs]


        plt.plot(xs, [i.real for i in ys1], label='')
        plt.plot(xs, [i.real for i in ys2], label='')

        """fig.add_trace(go.Scatter(x=xs, y=[i.real for i in ys1], line=dict(color="red")))
        fig.add_trace(go.Scatter3d(x=xs, y=[i.real for i in ys2], line=dict(color="green")))"""
plt.show()


'''rect1 = (-0.2, -0.1, 2.2, 2.3)
rect2 = (-0.8, -1.2, 0, 1)
rect = (-1, -0.5, -1, 0)


def zeidel(funcs, eps=0.000001):
    x_ss = [Symbol(f'x{i}') for i in range(len(funcs))]
    xs = [solve(funcs, x)[0][0] for func, x in zip(funcs, x_ss)]
    yacobi = [[diff(fi, x) for fi in xs] for x in x_ss]
    for x in np.arange(rect[0], rect[1], abs(rect[0] - rect[1]) / 10):
        for y in np.arange(rect[2], rect[3], abs(rect[2] - rect[3]) / 10):
            yac = np.array([[float(el.subs(x_ss[0], x).subs(x_ss[1], y)) for el in line] for line in yacobi])
            if max([sum(map(abs, i)) for i in yac]) >= 1:                print('Error')
                return
    args = [rect[0], rect[2]]
    las_args = [rect[0], rect[2]]
    while True:
        for i in range(len(args)):
            args[i] = N(xs[i].subs(x_ss[0], args[0]).subs(x_ss[1], args[1]))
        mdiff = sqrt((args[0] - las_args[0]) ** 2 + (args[1] - las_args[1]) ** 2)
        if mdiff < eps:
            break
        las_args = args.copy()
    return args


print("Метод Зейделя: ", zeidel((function_system_first, function_system_second)))'''


def grad_descent(funcs, eps=0.00001, alpha=0.1):
    M = sum(i**2 for i in funcs)
    xSs = [Symbol(f'x{i}') for i in range(len(funcs))]
    grad = [diff(M, x) for x in xSs]
    x = [1, 1]
    x_p = [1, 1]
    while True:
        x = x_p - alpha * np.array([N(grad[i].subs(xSs[0], x_p[0]).subs(xSs[1], x_p[1])) for i in range(len(grad))])
        mdiff = sqrt((x[0] - x_p[0]) ** 2 + (x[1] - x_p[1]) ** 2)
        if mdiff < eps:
            break
        x_p = x
    return x


print("Метод Градиентного спуска: ", grad_descent((function_system_first, function_system_second)))

"""
const int n = 5; // число точек
double X[n] = {0.0, 1.0, 2.0, 3.0, 4.0}; // x координаты точек
double Y[n] = {1.0, 4.0, 2.0, 5.0, 3.0}; // y координаты точек

double Stirling[n][n]; // многочлен Стирлинга
double h = X[1] - X[0]; // шаг на сетке

// вычисляет многочлен Стирлинга для заданных точек
void calc_stirling() {
    // инициализируем первый столбец
     for (int i = 0; i < n; i++) {
        Stirling[i][i] = Y[i];
    }

    // заполняем таблицу коэффициентов многочлена
    for (int j = 1; j < n; j++) {
        for (int i = j; i < n; i++) {
            Stirling[i][j] = Stirling[i][j-1] - Stirling[i-1][j-1];
        }
    }
}

// вычисляет значение многочлена Стирлинга в точке x
double k(int j, double x) {
    double t = (x - X[0])/h;
    double res = 0;
    double p = 1;

    for (int i = 0; i <= j; i++) {
        res += Stirling[i][j]*p;
        p *= (t - i + 1)/i;
    }
    return res;
}

// вычисляет значение интерполяционного многочлена в точке x
double interpolate(double x) {
    double res = 0;
    for (int j = 0; j < n; j++) {
        res += k(j, x);
    }
    return res;
}

int main() {
    calc_stirling(); // вычисляем многочлен Стирлинга
    double x = 2.5; // точка, в которой мы будем интерполировать функцию
    double y = interpolate(x); // вычисляем значение интерполяционного многочлена в точке x
    cout << "Значение функции в точке " << x << " равно " << y << endl;
    return 0;
}
"""
