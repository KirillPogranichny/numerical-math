import numpy as np
import warnings


warnings.simplefilter(action="ignore", category=RuntimeWarning)


def iterate_method(_a, _n):
    eps = 10e-4
    x_prev = np.full(_n, 1.)
    lambda_prev = -1
    k = 0
    while True:
        x_next = _a.dot(x_prev)
        lambda_next = max(x_next[i] / x_prev[i] for i in range(_n))
        k += 1
        if np.linalg.norm(lambda_next - lambda_prev) <= eps:
            return lambda_next, k
        lambda_prev = lambda_next
        x_prev = x_next


if __name__ == '__main__':
    a3 = np.array([
        [5, 2, 1],
        [2, 4, 2],
        [1, 2, 9]
    ])

    a5 = np.array([
        [10, 1, 2, 3, 1],
        [1, 2, -3, 1, 3],
        [2, -3, 5, 7, 2],
        [3, 1, 7, 9, 1],
        [1, 3, 2, 1, 10]
    ])

    a7 = np.array([
        [10, 1, 2, 3, 2, 1, 1],
        [1, 4, -3, 1, 1, 1, 1],
        [2, -3, 3, 2, 1, 1, 2],
        [3, 1, 2, 10, 1, 1, 3],
        [2, 1, 1, 1, 5, 1, 2],
        [1, 1, 1, 1, 1, 4, 1],
        [1, 1, 2, 3, 2, 1, 10]
    ])

    print("a:")
    print(a3)
    print("a:")
    print(a5)
    print("a:")
    print(a7)

    print("n = 3:")
    print("непосредственное решение: ", max(np.linalg.eigvals(a3)))
    answer, k = iterate_method(a3, len(a3))
    print("решение, полученное методом итераций: ", answer)
    print("количество шагов: ", k)

    print("n = 5:")
    print("непосредственное решение: ", max(np.linalg.eigvals(a5)))
    answer, k = iterate_method(a5, len(a5))
    print("решение, полученное методом итераций: ", answer)
    print("количество шагов: ", k)

    print("n = 7:")
    print("непосредственное решение: ", max(np.linalg.eigvals(a7)))
    answer, k = iterate_method(a7, len(a7))
    print("решение, полученное методом итераций: ", answer)
    print("количество шагов: ", k)
