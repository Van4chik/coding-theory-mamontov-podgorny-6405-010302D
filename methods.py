import numpy as np
from itertools import combinations

# 1.1
def ref(matrix):

    # преобразование матрицы в массив numpy
    mat = np.array(matrix)
    n_rows, n_cols = mat.shape
    lead = 0
    for r in range(n_rows):
        if lead >= n_cols:
            return mat
        i = r
        while mat[i, lead] == 0:
            i += 1
            if i == n_rows:
                i, lead = r, lead + 1
                if lead == n_cols:
                    return mat

        # если необходимо, то меняем строки местами
        mat[[i, r]] = mat[[r, i]]

        # обработка всех строк ниже текущей
        for i in range(r + 1, n_rows):
            if mat[i, lead] != 0:
                mat[i] = (mat[i] + mat[r]) % 2
        lead += 1
    return mat


# 1.2
def rref(mat):
    mat = ref(mat)
    n_rows, n_cols = mat.shape

    # проход по строкам сверху вниз
    for r in range(n_rows - 1, -1, -1):

        # находим ведущий элемент в строке
        lead = np.argmax(mat[r] != 0)
        if mat[r, lead] != 0:

            # обнуляем все элементы, которые выше ведущего
            for i in range(r - 1, -1, -1):
                if mat[i, lead] != 0:
                    mat[i] = (mat[i] + mat[r]) % 2
    while not any(mat[n_rows - 1]):
        mat, n_rows = mat[:-1, :], n_rows - 1
    return mat


# 1.3: ведущие столбцы и удаление их для создания сокращённой матрицы
def find_lead_columns(matrix):
    lead_columns = []
    for r in range(len(matrix)):
        row = matrix[r]
        for i, val in enumerate(row):
            if val == 1:  # первый элемент 1 в строке - ведущий
                lead_columns.append(i)
                break
    return lead_columns


# функция, удаляющая ведущие столбцы
def remove_lead_columns(matrix, lead_columns):
    mat = np.array(matrix)
    reduced_matrix = np.delete(mat, lead_columns, axis = 1)
    return reduced_matrix


# Шаг 4: формирование матрицы H
def form_H_matrix(X, lead_columns, n_cols):

    # инициализация единичной матрицы размером (n - k) на n
    n_rows = np.shape(X)[1]

    H = np.zeros((n_cols, n_rows), dtype = int)
    I = np.eye(6, dtype = int)

    H[lead_columns, :] = X
    not_lead = [i for i in range(n_cols) if i not in lead_columns]
    H[not_lead, :] = I

    return H


# основная функция для выполнения всех шагов
def LinearCode(mat):

    # 1.3.1: преобразование матрицы в ступенчатый вид
    g_star = rref(mat)

    print("G* (RREF матрица) =")
    print(g_star)

    # 1.3.2: найти ведущие столбцы
    lead_columns = find_lead_columns(g_star)
    print(f"lead = {lead_columns}")

    # 1.3.3: удалить ведущие столбцы и получить сокращённую матрицу
    X = remove_lead_columns(g_star, lead_columns)
    print("Сокращённая матрица X =")
    print(X)

    # 1.3.4: сформировать проверочную матрицу H
    n_cols = np.shape(mat)[1]
    H = form_H_matrix(X, lead_columns, n_cols)
    print("Проверочная матрица H =")
    print(H)

    return H


# 1.3

# функция для нахождения всех кодовых слов из порождающей матрицы
def generate_codewords_from_combinations(G):
    rows = G.shape[0]
    codewords = set()

    # перебираем все возможные комбинации строк матрицы G
    for r in range(1, rows + 1):
        for comb in combinations(range(rows), r):

            # суммируем строки и добавляем результат в множество
            codeword = np.bitwise_xor.reduce(G[list(comb)], axis = 0)
            codewords.add(tuple(codeword))

    # добавляем в множество нулевой вектор
    codewords.add(tuple(np.zeros(G.shape[1], dtype = int)))

    return np.array(list(codewords))

# функция для умножения всех двоичных слов длины k на G
def generate_codewords_binary_multiplication(G):
    k = G.shape[0]
    n = G.shape[1]
    codewords = []

    # генерируем все двоичные слова длины k
    for i in range(2**k):
        binary_word = np.array(list(np.binary_repr(i, k)), dtype = int)
        codeword = np.dot(binary_word, G) % 2
        codewords.append(codeword)

    return np.array(codewords)

# функция проверки кодового слова с помощью проверочной матрицы H
def check_codeword(codeword, H):
    return np.dot(codeword, H) % 2

# функция вычисления кодового расстояния
def calculate_code_distance(codewords):
    min_distance = float('inf')

    # подсчет количества ненулевых элементов для всех попарных разностей кодовых слов
    for i in range(len(codewords)):
        for j in range(i + 1, len(codewords)):
            distance = np.sum(np.bitwise_xor(codewords[i], codewords[j]))
            if distance > 0:
                min_distance = min(min_distance, distance)

    return min_distance

# основная функция для выполнения всех шагов
def LinearCodeWithErrors(mat):

    # выполнение шагов, как и ранее
    g_star = rref(mat)
    lead_columns = find_lead_columns(g_star)
    X = remove_lead_columns(g_star, lead_columns)
    n_cols = np.shape(mat)[1]
    H = form_H_matrix(X, lead_columns, n_cols)

    print("G* (RREF матрица) =")
    print(f"{g_star}\n")
    print(f"lead = {lead_columns}\n")
    print("Сокращённая матрица X =")
    print(f"{X}\n")
    print("Проверочная матрица H =")
    print(f"{H}\n")

    # 1.4.1: генерация всех кодовых слов через сложение строк
    codewords_1 = generate_codewords_from_combinations(g_star)
    print("Все кодовые слова (способ 1):")
    print(f"{codewords_1}\n")

    # 1.4.2: генерация кодовых слов умножением двоичных слов на G
    codewords_2 = generate_codewords_binary_multiplication(g_star)
    print("Все кодовые слова (способ 2):")
    print(f"{codewords_2}\n")

    # проверка, что множества кодовых слов совпадают
    assert set(map(tuple, codewords_1)) == set(map(tuple, codewords_2)), "Наборы кодовых слов не совпадают!"

    # проверка кодовых слов с помощью матрицы H
    for codeword in codewords_1:
        result = check_codeword(codeword, H)
        assert np.all(result == 0), f"Ошибка: кодовое слово {codeword} не прошло проверку матрицей H"

    print("Все кодовые слова прошли проверку матрицей H.")

    # 1.4: вычисление кодового расстояния
    d = calculate_code_distance(codewords_1)
    t = 0
    if t == 0:
        t = 1
    else:
        t = (d - 1) // 2
    print(f"Кодовое расстояние d = {d}")
    print(f"Кратность обнаруживаемой ошибки t = {t}\n")

    # проверка ошибки кратности t
    e1 = np.zeros(n_cols, dtype = int)
    e1[2] = 1                            # внесение ошибки в один бит
    v = codewords_1[4]
    print(f"e1 = {e1}")
    print(f"v = {v}")
    v_e1 = (v + e1) % 2
    print(f"v + e1 = {v_e1}")
    print(f"(v + e1)@H = {check_codeword(v_e1, H)} - error\n")

    # проверка ошибки кратности t + 1
    e2 = np.zeros(n_cols, dtype = int)
    e2[6] = 1
    e2[9] = 1                            # внесение ошибки в два бита
    print(f"e2 = {e2}")
    v_e2 = (v + e2) % 2
    print(f"v + e2 = {v_e2}")
    print(f"(v + e2)@H = {check_codeword(v_e2, H)} - no error")

    return H