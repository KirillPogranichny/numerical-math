import numpy as np


def initial_N(dim):
    N = np.add(-np.triu(np.ones((dim, dim), dtype=int), 1), np.tri(dim, dim, 0, dtype=int))
    return N


def initial_A(dim):
    A = np.add(-np.triu(np.ones((dim, dim), dtype=int), 1), np.identity(dim, dtype=int))
    return A


def initial_b(dim):
    b = -np.ones(dim, dtype=int)
    b[dim - 1] = 1
    return b


def solve(e, N, A, b):
    x = np.linalg.tensorsolve(np.add(A, (pow(10, e) * N)), b)
    m = np.linalg.cond(A + pow(10, e) * N)
    print("\nДля e =", e)
    print("x =", x)
    print("μ =", m)


if __name__ == "__main__":
    dim = int(input("Введите размерность матрицы: "))

    N = initial_N(dim)
    print("N:\n", N)
    A = initial_A(dim)
    print("\nA:\n", A)
    b = initial_b(dim)
    print("\nb:\n", b)

    for e in range(-6, -2): solve(e, N, A, b)
