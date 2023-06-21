import math
import numpy as np


def relaxation(_A, _b, _x_start):
    omega = 0.7
    _x_next = np.copy(_x_start)
    for i in range(len(_A)):
        _x_next[i] = (1 - omega) * _x_start[i] +\
                     omega * (- sum((_A[i][j] * _x_start[j]) for j in range(i + 1, len(_A)))
                              - sum((_A[i][j] * _x_next[j]) for j in range(0, i)) + _b[i]) / _A[i][i]
    k = 1
    while np.linalg.norm(_x_next - _x_start) >= 10e-10:
        _x_start = np.copy(_x_next)
        for i in range(len(_A)):
            _x_next[i] = (1 - omega) * _x_start[i] +\
                        omega * (- sum((_A[i][j] * _x_start[j]) for j in range(i + 1, len(_A)))
                                 - sum((_A[i][j] * _x_next[j]) for j in range(0, i)) + _b[i]) / _A[i][i]

        k += 1

    return _x_next, k


def nu(k, n):
    return math.cos((2 * k - 1) * math.pi / (2 * n))


def richardson(_A, _b, _x_start):
    lambda_max = max(np.linalg.eigvals(_A))
    lambda_min = min(np.linalg.eigvals(_A))

    eta = lambda_min / lambda_max
    ro0 = (1 - eta) / (1 + eta)
    tau0 = 2 / (lambda_max + lambda_min)
    n = 7

    _x_next = _x_start + (tau0 / (1 + ro0 * nu(1, n))) * (- _A.dot(_x_start) + _b)
    k = 2
    while k < n and np.linalg.norm(_x_next - _x_start) >= 10e-10:
        _x_start = np.copy(_x_next)
        _x_next = _x_start + (tau0 / (1 + ro0 * nu(k, n))) * (- _A.dot(_x_start) + _b)
        k += 1

    return _x_next, k


if __name__ == "__main__":
    A = np.array([
        [4, -1, -6, 0],
        [-5, -4, 10, 8],
        [0, 9, 4, -2],
        [1, 0, -7, 5]
    ])
    b = np.array([2, 21, -12, -6])

    print(np.linalg.eigvals(A))
    x_start = np.array([0.00, 0.00, 0.00, 0.00])

    x1, k1 = relaxation(A, b, x_start)
    print(f"Метод последовательной верхней релаксации:\nx = {x1}\nk = {k1}\n")
    x2, k2 = richardson(A, b, x_start)
    print(f"Метод Ричардсона:\nx = {x2}\nk = {k2}\n")

    print(f"Действительные значения x = {np.linalg.inv(A).dot(b)}")
