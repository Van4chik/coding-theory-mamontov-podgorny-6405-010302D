import numpy as np
import random

# Функция для нахождения допущенной ошибки
def get_error(w, H, B):
    s = w @ H % 2
    u1 = None
    if sum(s) <= 3:
        u1 = np.array(s)
        u1 = np.hstack((u1, np.zeros(len(s), dtype=int)))
    else:
        for i in range(len(B)):
            temp = (s + B[i]) % 2
            if sum(temp) <= 2:
                ei = np.zeros(len(s), dtype=int)
                ei[i] = 1
                u1 = np.hstack((temp, ei))
    if u1 is not None:
        u1
    else:
        sB = s @ B % 2
        if sum(sB) <= 3:
            u1 = np.hstack((np.zeros(len(s), dtype=int), sB))
        else:
            for i in range(len(B)):
                temp = (sB + B[i]) % 2
                if sum(temp) <= 2:
                    ei = np.zeros(len(s), dtype=int)
                    ei[i] = 1
                    u1 = np.hstack((ei, temp))
    return u1

# Функция для допущения ошибки и поиска этой самой ошибки при помощи get_error()
def gen_and_check_error(u, G, H, B, error_rate):
    print("Исходное сообщение:", u)
    w = u @ G % 2
    print("Отправленное сообщение", w)
    error = np.zeros(w.shape[0], dtype=int)
    error_indices = random.sample(range(w.shape[0]), error_rate)
    for index in error_indices:
        error[index] = 1
    print("Допущенная ошибка:", error)
    w = (w + error) % 2
    print("Сообщение с ошибкой", w)
    error = get_error(w, H, B)
    print("Вызываем get_error, получаем ошибку:", error)
    if error is None:
        print("Ошибка обнаружена, исправить невозможно!")
        return
    message = (w + error) % 2
    print("Исправленное отправленное сообщение:", message)
    w = u @ G % 2
    if (not np.array_equal(w, message)):
        print("Сообщение было декодировано с ошибкой!")

# Функция для формирования порождающей матрицы кода Рида-Маллера
def reed_muller_generator_matrix(r: int, m: int) -> np.ndarray:
    # Базовый случай: r = 0 -> вектор из 1 длины 2^m
    if r == 0:
        return np.ones((1, 2 ** m), dtype=int)

    # Базовый случай: r = m -> G(m-1, m) и внизу вектор [0...01]
    if r == m:
        G_m_m_1_m = reed_muller_generator_matrix(m - 1, m)
        bottom_row = np.zeros((1, 2 ** m), dtype=int)
        bottom_row[0, -1] = 1
        return np.vstack([G_m_m_1_m, bottom_row])

    # Рекурсивный случай: [[G(r, m-1), G(r, m-1)],[0, G(r-1, m-1)]]
    G_r_m_m_1 = reed_muller_generator_matrix(r, m - 1)
    G_r_m_1_m_m_1 = reed_muller_generator_matrix(r - 1, m - 1)

    # Верхняя часть: G(r, m-1) дублируется
    top = np.hstack([G_r_m_m_1, G_r_m_m_1])

    # Нижняя часть: нули слева и G(r-1, m-1) справа
    bottom = np.hstack([np.zeros((G_r_m_1_m_m_1.shape[0], G_r_m_1_m_m_1.shape[1]), dtype=int), G_r_m_1_m_m_1])

    # Объединение верхней и нижней части
    return np.vstack([top, bottom])

# Функция для произведения Кронекера
def kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Получаем размеры матриц
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    # Инициализируем результирующую матрицу
    result = np.zeros((rows_A * rows_B, cols_A * cols_B), dtype=A.dtype)

    # Вычисляем произведение Кронекера
    for i in range(rows_A):
        for j in range(cols_A):
            result[i * rows_B:(i + 1) * rows_B, j * cols_B:(j + 1) * cols_B] = A[i, j] * B

    return result

# Функция для формирования H_m^i
def H_matrix(H, m, i):
    matrix = np.eye(2 ** (m - i), dtype=int)
    matrix = kronecker_product(matrix, H)
    matrix = kronecker_product(matrix, np.eye(2 ** (i - 1)))
    return matrix

# Функция, которая допускает ошибку в отправленном сообщении и исправляет ее
def gen_and_check_error_RM(u, G, error_rate, m):
    print("Исходное сообщение:", u)
    w = u @ G % 2
    print("Отправленное сообщение", w)
    error = np.zeros(w.shape[0], dtype=int)
    error_indices = random.sample(range(w.shape[0]), error_rate)
    for index in error_indices:
        error[index] = 1
    print("Допущенная ошибка:", error)
    w = (w + error) % 2
    print("Сообщение с ошибкой", w)
    for i in range(len(w)):
        if w[i] == 0:
            w[i] = -1
    w_array = []
    H = np.array([[1, 1], [1, -1]])
    w_array.append(w @ H_matrix(H, m, 1))
    for i in range(2, m + 1):
        w_array.append(w_array[-1] @ H_matrix(H, m, i))
    maximum = w_array[0][0]
    index = -1
    for i in range(len(w_array)):
        for j in range(len(w_array[i])):
            if abs(w_array[i][j]) > abs(maximum):
                index = j
                maximum = w_array[i][j]
    counter = 0
    for i in range(len(w_array)):
        for j in range(len(w_array[i])):
            if abs(w_array[i][j]) == abs(maximum):
                counter += 1
            if (counter > 1):
                print("Невозможно исправить ошибку!")
                return
    message = list(map(int, list(('{' + f'0:0{m}b' + '}').format(index))))
    if maximum > 0:
        message.append(1)
    else:
        message.append(0)
    print("Исправленное сообщение:", np.array(message[::-1]))
    if (not np.array_equal(u, message)):
        print("Сообщение было декодировано с ошибкой!")