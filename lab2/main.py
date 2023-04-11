import numpy as np


def initial():
    A1 = np.array([
        [13.14, -2.12, 1.17],
        [-2.12, 6.3, -2.45],
        [1.17, -2.45, 4.6]
    ])
    b1 = np.array([1.27, 2.13, 3.14])

    A2 = np.array([
        [4.31, 0.26, 0.61, 0.27],
        [0.26, 2.32, 0.18, 0.34],
        [0.61, 0.18, 3.20, 0.31],
        [0.27, 0.34, 0.31, 5.17]
    ])
    b2 = np.array([1.02, 1.00, 1.34, 1.27])

    return A1, b1, A2, b2


def LU_decomposition(A):
    dim = np.shape(A)[0]
    U = np.zeros(np.shape(A))
    U += A
    L = np.add(np.zeros((dim, dim), dtype=float), np.identity(dim, dtype=float))

    for k in range(1, dim):
        for i in range(k-1, dim):
            for j in range(i, dim):
                L[j][i] = U[j][i] / U[i][i]

        for i in range(k, dim):
            for j in range(k-1, dim):
                U[i][j] = U[i][j] - L[i][k-1] * U[k-1][j]

    return L, U


def solve_SLAE(L, U, b):
    y = np.linalg.tensorsolve(L, b)
    x = np.linalg.tensorsolve(U, y)
    return x


if __name__ == "__main__":
    A1, b1, A2, b2 = initial()

    L, U = LU_decomposition(A1)
    print(f"Разложение для матрицы A:\n {A1}")
    print(f"\nи вектора b:\n {b1}")
    print("\nL:\n", L)
    print("\nU:\n", U)
    x = solve_SLAE(L, U, b1)
    print("\nРешение для x:", x)

    L, U = LU_decomposition(A2)
    print(f"\n\nРазложение для матрицы A:\n {A2}")
    print(f"\nи вектора b:\n {b2}")
    print("\nL:\n", L)
    print("\nU:\n", U)
    x = solve_SLAE(L, U, b2)
    print("\nРешение для x:", x)