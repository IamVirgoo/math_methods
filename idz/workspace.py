import random
import matplotlib.pyplot as plt


"""
    Реализация формул Ньютона-Стирлинга и Ньютона-Бесселя
"""


number = int(input('Enter the number of array`s: '))
start, end = int(input('Enter the start number: ')), int(input('Enter the end number: '))
x_array = [random.randint(start, end) for _ in range(number)]
y_array = [random.randint(start, end) for _ in range(number)]
stirling = [[0] * number for i in range(number)]
step_on_grid = x_array[1] - x_array[0]


def calculate_stirling():
    for i in range(number):
        stirling[i][i] = y_array[i]
    for j in range(number):
        for i in range(number):
            stirling[i][j] = stirling[i][j - 1] - stirling[i - 1][j - 1]


def calculate_in_dot(j, x_dot):
    t = (x_dot - x_array[0]) / step_on_grid
    result = 0
    proizv = 1
    for i in range(1, j):
        result += stirling[i][j] * proizv
        proizv *= (t - i + 1) / i
    return result


def interpolate(x_dot):
    result = 0
    for j in range(number):
        result += calculate_in_dot(j, x_dot)
    return result


calculate_stirling()
print(f'\nЗначения x: {x_array} \nЗначения y: {y_array}\n')
x_dot = float(input('Enter the x_dot number: '))
y_dot = interpolate(x_dot)
plt.plot(x_array, y_array)
plt.show()
print(f'Значение функции в точке {x_dot} равно {y_dot}')
