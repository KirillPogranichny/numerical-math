import numpy as np


def proj(x, y):
    return y * (x.dot(y) / y.dot(y))


def qr_decomposition(A):
    dim = A.shape
    R = np.zeros((dim[1], dim[1]))
    U = np.zeros(dim)
    C = np.zeros(dim)

    for k in range(dim[0]):
        sum = 0

        if k > 0:
            sum = 0
            for j in range(k-1):
                sum += proj(A.transpose()[k], U[j])

        U[k] = A.transpose()[k] - sum
        C[k] = U[k] / np.linalg.norm(U[k])

    for i in range(dim[1]):
        for j in range(dim[1]):
            if i <= j:
                R[i][j] = C[i].dot(A.transpose()[j])

    Q = C.T

    return Q, R


def get_solution(A, b):
    Q, R = qr_decomposition(A)
    y = np.linalg.tensorsolve(Q, b)

    return np.linalg.tensorsolve(R, y)


if __name__ == "__main__":
    A = np.array([
        [2, 0, 1],
        [0, 1, -1],
        [1, 1, 1]
    ])
    b = np.array([3, 0, 3])

    Q, R = qr_decomposition(A)
    print("Решение системы:\n", get_solution(A, b))
    print("\nТочное решение:\n", np.linalg.tensorsolve(A, b))
