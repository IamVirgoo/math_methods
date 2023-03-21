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


def pretty_print(first_matrix, second_matrix, selected):
    for row in range(len(second_matrix)):
        print("[", end='')
        for col in range(len(first_matrix[row])):
            print("\t{1:5f}{0}".format(" " if (selected is None
                                                 or selected != (row, col)) else "*", first_matrix[row][col]), end='')
        print("\t] * [X{0}] = [{1:5f}]".format(row + 1, second_matrix[row]))


# --- перемена местами двух строк системы
def swap_rows(first_matrix, second_matrix, row1, row2):
    first_matrix[row1], first_matrix[row2] = first_matrix[row2], first_matrix[row1]
    second_matrix[row1], second_matrix[row2] = second_matrix[row2], second_matrix[row1]


# --- деление строки системы на число
def divide_row(first_matrix, second_matrix, row, divider):
    first_matrix[row] = [a / divider for a in first_matrix[row]]
    second_matrix[row] /= divider


# --- сложение строки системы с другой строкой, умноженной на число
def combine_rows(first_matrix, second_matrix, row, source_row, weight):
    first_matrix[row] = [(a + k * weight) for a, k in zip(first_matrix[row], first_matrix[source_row])]
    second_matrix[row] += second_matrix[source_row] * weight


# --- решение системы методом Гаусса (приведением к треугольному виду)
def gauss(first_matrix, seconds_matrix):
    column = 0
    while column < len(seconds_matrix):
        print("\nИщем максимальный по модулю элемент в {0}-м столбце:\n".format(column + 1))
        current_row = None
        for r in range(column, len(first_matrix)):
            if current_row is None or abs(first_matrix[r][column]) > abs(first_matrix[current_row][column]):
                current_row = r
        if current_row is None:
            print("\nрешений нет\n")
            return None
        pretty_print(first_matrix, seconds_matrix, (current_row, column))
        if current_row != column:
            print("\nПереставляем строку с найденным элементом повыше:\n")
            swap_rows(first_matrix, seconds_matrix, current_row, column)
            pretty_print(first_matrix, seconds_matrix, (column, column))
        print("\nНормализуем строку с найденным элементом:\n")
        divide_row(first_matrix, seconds_matrix, column, first_matrix[column][column])
        pretty_print(first_matrix, seconds_matrix, (column, column))
        print("\nОбрабатываем нижележащие строки:\n")
        for r in range(column + 1, len(first_matrix)):
            combine_rows(first_matrix, seconds_matrix, r, column, -first_matrix[r][column])
        pretty_print(first_matrix, seconds_matrix, (column, column))
        column += 1
    print("\nМатрица приведена к треугольному виду, считаем решение\n")
    X = [0 for b in seconds_matrix]
    for i in range(len(seconds_matrix) - 1, -1, -1):
        X[i] = seconds_matrix[i] - sum(x * a for x, a in zip(X[(i + 1):], first_matrix[i][(i + 1):]))
    print("\nПолучили ответ:\n")
    print("\n".join("X{0} =\t{1:5f}".format(i + 1, x) for i, x in
                    enumerate(X)))
    return X


print("Исходная система:")
pretty_print(A, B, None)
print("Решаем:")
gauss(A, B)
