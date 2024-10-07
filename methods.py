import numpy as np
import itertools
import random

# Функция формирования проверочной матрицы кода Хэмминга
def generate_hamming_h_matrix(r: int) -> np.ndarray:
    n = 2 ** r - 1  # Число строк и столбцов
    res = []
    cur_r = r - 1
    for i in range(n, 0, -1):
        if i != 2 ** cur_r:
            res.append(list(map(int, f"{i:0{r}b}")))
        else:
            cur_r -= 1

    # Единичная матрица r x r в нижней части
    identity_matrix = np.eye(r, dtype=int)

    # Объединяем верхнюю часть и единичную матрицу
    H = np.vstack((res, identity_matrix))

    return H

# Функция для построения порождающей матрицы кода Хэмминга на основе проверочной матрицы
def H_to_G(H: np.ndarray, r) -> np.ndarray:
    k = 2 ** r - r - 1
    res = np.eye(k, dtype=int)  # Единичная матрица
    G = np.hstack((res, H[:k]))
    return G

# Функция для построения таблицы синдромов
def generate_syndrome_table(matrix: np.ndarray, error_weight: int) -> dict:
    n = matrix.shape[0]
    syndrome_table = {}
    for error in range(1, error_weight + 1):
        for error_indices in itertools.combinations(range(n), error):
            error_vector = np.zeros(n, dtype=int)
            for index in error_indices:
                error_vector[index] = 1
            syndrome = error_vector @ matrix % 2
            syndrome_table[tuple(map(int, syndrome))] = tuple(error_indices)

    return syndrome_table

# Функция для допущения и проверки ошибки
def hamming_correction_test(G: np.ndarray, H: np.ndarray, syndrome_table: dict, error_degree: int, u: np.ndarray):
    # Шаг 1: Выбираем случайное кодовое слово из G
    print("Кодовое слово (u):", u)

    # Шаг 2: Генерируем кодовое слово
    v = u @ G % 2
    print("Отправленное кодовое слово (v):", v)

    # Шаг 3: Допускаем ошибку error_degree в принятом кодовом слове
    error = np.zeros(v.shape[0], dtype=int)
    error_indices = random.sample(range(v.shape[0]), error_degree)  # случайные индексы ошибок
    for index in error_indices:
        error[index] = 1
    print("Допущенная ошибка:", error)

    # Принятое слово с ошибкой
    received_v = (v + error) % 2
    print("Принятое с ошибкой слово:", received_v)

    # Шаг 4: Вычисляем синдром принятого слова
    syndrome = received_v @ H % 2
    print("Синдром принятого сообщения:", syndrome)
    if sum(syndrome) != 0:
        print("Обнаружена ошибка!")

    # Шаг 5: Проверяем синдром в таблице синдромов и корректируем ошибку
    if tuple(syndrome) in syndrome_table:
        correction_indices = syndrome_table[tuple(syndrome)]
        for index in correction_indices:
            received_v[index] = (received_v[index] + 1) % 2  # корректируем ошибку
        print("Исправленное сообщение:", received_v)

        # Проверка совпадения с отправленным сообщением
        if np.array_equal(v, received_v):
            print("Ошибка была исправлена успешно!")
        else:
            print("Ошибка не была исправлена корректно.")
    else:
        print("Синдрома нет в таблице, ошибка не исправлена.")

def expand_G_matrix(G: np.ndarray) -> np.ndarray:
    col = np.zeros((G.shape[0], 1), dtype=int)
    for i in range(G.shape[0]):
        if sum(G[i]) % 2 == 1:
            col[i] = 1
    return np.hstack((G, col))