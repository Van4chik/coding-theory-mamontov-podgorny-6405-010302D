import numpy as np
import methods

if __name__ == '__main__':
    print("4.1\n")
    print("Зададим матрицу B для расширенного кода Голея")
    B = np.array([[1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                  [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                  [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                  [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                  [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                  [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                  [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                  [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                  [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                  [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                  [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    print(B)

    print("\nПостроим порождающую матрицу")
    G = np.hstack((np.eye(12, 12, dtype=int), B))
    print(G)

    print("\nПостроим проверочную матрицу")
    H = np.vstack((np.eye(12, 12, dtype=int), B))
    print(H)

    print("\n4.2\n")
    print("Отправляем сообщение (1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0). Допустим однократную ошибку\n")
    u = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    methods.gen_and_check_error(u, G, H, B, 1)
    print("\nДопустим двухкратную ошибку\n")
    methods.gen_and_check_error(u, G, H, B, 2)
    print("\nДопустим трехкратную ошибку\n")
    methods.gen_and_check_error(u, G, H, B, 3)
    print("\nДопустим четырехкратную ошибку\n")
    methods.gen_and_check_error(u, G, H, B, 4)
    print("\nЧасть 2\n")
    print("4.3\n")
    print("Введена функция для формирования порождающей матрицы кода Рида-Маллера\n")
    print("4.4\n")
    print("Сформируем порождающую матрицу для RM(1, 3)")
    G = methods.reed_muller_generator_matrix(1, 3)
    print(G)
    print("\nДопустим однократную ошибку\n")
    m = 3
    u = np.array([1, 0, 0, 1])
    methods.gen_and_check_error_RM(u, G, 1, m)
    print("\nДопустим двухкратную ошибку\n")
    methods.gen_and_check_error_RM(u, G, 2, m)
    print("\n4.5\n")
    print("Сформируем порождающую матрицу для RM(1, 4)")
    G = methods.reed_muller_generator_matrix(1, 4)
    print(G)
    print("\nДопустим однократную ошибку\n")
    m = 4
    u = np.array([1, 0, 1, 0, 1])
    methods.gen_and_check_error_RM(u, G, 1, m)
    print("\nДопустим двухкратную ошибку\n")
    methods.gen_and_check_error_RM(u, G, 2, m)
    print("\nДопустим трехкратную ошибку\n")
    methods.gen_and_check_error_RM(u, G, 3, m)
    print("\nДопустим четырехкратную ошибку\n")
    methods.gen_and_check_error_RM(u, G, 4, m)