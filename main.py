import methods

if __name__ == '__main__':

    # Пример
    matrix = ([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
               [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
               [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
               [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
               [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
               [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
    )
    methods.LinearCodeWithErrors(matrix)