import random
import methods
import numpy as np

if __name__ == "__main__":
    print("Часть 1")
    s_matrix = np.array([[1, 0, 0, 1, 0, 1, 1],
                         [1, 1, 0, 0, 0, 0, 1],
                         [0, 0, 1, 1, 0, 0, 1],
                         [1, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 1, 1, 1, 0]])
    G = methods.RREF(methods.REF(s_matrix))
    print("Порождающая матрица G:", G, sep = "\n")
    G_standard = methods.standard_view(G)
    print("Порождающая матрица G в стандартном виде:", G_standard, sep = "\n")
    print()

    H = methods.h_matrix(G_standard)
    print("Проверочная матрица H:", H, sep = "\n")
    print()

    syndrome_table = methods.generate_syndrome_table(H, 1)
    print("Таблица синдромов:", syndrome_table, sep = "\n")
    print()

    u = np.array([1, 0, 0, 1])
    print("Кодовое слово длины k = 4:", u, sep = "\n")
    v = u @ G_standard % 2
    print("Отправленное кодовое слово длины n = 7:", v, sep = "\n")
    error = np.array([0] * 7)
    error[random.randint(0, 6)] = 1
    print("Возникшая ошибка:", error, sep = "\n")
    v = (v + error) % 2
    print("Принятое с ошибкой слово:", v, sep = "\n")
    syndrome = v @ H % 2
    print("Синдром принятого сообщения:", syndrome, sep = "\n")
    error = np.array([0] * 7)
    error[syndrome_table[tuple(syndrome)][0]] = 1
    v = (v + error) % 2
    print("Исправленное сообщение:", v, sep = "\n")
    print("Отправленное и исправленное сообщение совпадают")
    print()

    print("Кодовое слово длины k = 4:", u, sep = "\n")
    print("Отправленное кодовое слово длины n = 7:", v, sep = "\n")
    error = np.zeros(7, dtype = int)
    a, b = random.sample(range(7), 2)
    error[a], error[b] = 1, 1
    print("Возникшая ошибка:", error, sep = "\n")
    v = (v + error) % 2
    print("Принятое с ошибкой слово:", v, sep = "\n")
    syndrome = v @ H % 2
    print("Синдром принятого сообщения:", syndrome, sep = "\n")
    error = np.array([0] * 7)
    error[syndrome_table[tuple(syndrome)][0]] = 1
    v = (v + error) % 2
    print("Исправленное сообщение:", v, sep = "\n")
    print("Отправленное и исправленное сообщение не совпадают")
    print()


    print("2 часть")
    G_standard = np.array([[1,0,0,0,1,1,1,1,0,0,0,0],
                            [0,1,0,0,0,1,1,1,1,1,0,0],
                            [0,0,1,0,1,0,0,1,1,1,1,0],
                            [0,0,0,1,0,0,1,1,0,0,1,1]])
    print("Порождающая матрица G в стандартном виде:", G_standard, sep = "\n")
    print()

    H = methods.h_matrix(G_standard)
    print("Проверочная матрица H:", H, sep = "\n")
    print()

    syndrome_table = methods.generate_syndrome_table(H, 2)
    print("Таблица синдромов:", syndrome_table, sep = "\n")
    print()

    u = np.array([0, 0, 1, 0])
    print("Кодовое слово длины k = 4:", u, sep = "\n")
    v = u @ G_standard % 2
    print("Отправленное кодовое слово длины n = 12:", v, sep = "\n")
    error = np.array([0] * 12)
    error[random.randint(0, 11)] = 1
    print("Возникшая ошибка:", error, sep = "\n")
    v = (v + error) % 2
    print("Принятое с ошибкой слово:", v, sep = "\n")
    syndrome = v @ H % 2
    print("Синдром принятого сообщения:", syndrome, sep = "\n")
    error = np.array([0] * 12)
    for index in syndrome_table[tuple(syndrome)]:
        error[index] = 1
    v = (v + error) % 2
    print("Исправленное сообщение:", v, sep = "\n")
    print("Отправленное и исправленное сообщение совпадают")
    print()

    print("Кодовое слово длины k = 4:", u, sep = "\n")
    print("Отправленное кодовое слово длины n = 12:", v, sep = "\n")
    error = np.array([0] * 12)
    a, b = random.sample(range(12), 2)
    error[a], error[b] = 1, 1
    print("Возникшая ошибка:", error, sep = "\n")
    v = (v + error) % 2
    print("Принятое с ошибкой слово:", v, sep = "\n")
    syndrome = v @ H % 2
    print("Синдром принятого сообщения:", syndrome, sep = "\n")
    error = np.array([0] * 12)
    for index in syndrome_table[tuple(syndrome)]:
        error[index] = 1
    v = (v + error) % 2
    print("Исправленное сообщение:", v, sep = "\n")
    print("Отправленное и исправленное сообщение совпадают")
    print()

    print("Кодовое слово длины k = 4:", u, sep = "\n")
    print("Отправленное кодовое слово длины n = 12:", v, sep = "\n")
    error = np.array([0] * 12)
    a, b, c = random.sample(range(12), 3)
    error[a], error[b], error[c] = 1, 1, 1
    print("Возникшая ошибка:", error, sep = "\n")
    v = (v + error) % 2
    print("Принятое с ошибкой слово:", v, sep = "\n")
    syndrome = v @ H % 2
    print("Синдром принятого сообщения:", syndrome, sep = "\n")
    error = np.array([0] * 12)
    if tuple(syndrome) in syndrome_table:
        for index in syndrome_table[tuple(syndrome)]:
            error[index] = 1
        v = (v + error) % 2
        print("Исправленное сообщение:", v, sep = "\n")
        print("Отправленное и исправленное сообщение не совпадают")
        print()
    else:
        print("Синдрома, соответствующего данной ошибке, не найдено в таблице синдромов. Сообщение исправить невозможно.")