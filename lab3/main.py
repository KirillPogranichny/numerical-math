import numpy as np


class Solution:
    def __init__(self, a: np.array, b: np.array):
        self.dim = a.shape[0]
        self.b = b

        self.u = np.zeros(a.shape, dtype=float)

        self.u[0][0] = np.sqrt(a[0][0])

        for i in range(0, self.dim):
            for j in range(i):
                if j > 0:
                    self.u[0][j] = a[0][j] / self.u[0][0]
                sum = 0
                for k in range(i - 1):
                    sum += pow(self.u[k][i], 2)

                self.u[i][i] = np.sqrt(a[i][i] - sum)
            for j in range(i + 1, self.dim):
                if i < j:
                    sum = 0
                    for k in range(i - 1):
                        sum += self.u[k][i] * self.u[k][j]

                self.u[i][j] = ((a[i][j] - sum) / self.u[i][i])

    def get_solution(self):
        self.y = np.linalg.tensorsolve(self.u.T, self.b)
        return np.linalg.tensorsolve(self.u, self.y)

    def display_u_matrix(self):
        print(f"Матрица U:\n{self.u}")

    def display_solution(self):
        print(f"Решение системы:\n{self.get_solution()}\n")


if __name__ == "__main__":
    a_1 = np.array([
        [5.8, 0.3, 0.2],
        [0.3, 4, 0.7],
        [0.2, 0.7, 6.7]
    ])
    b_1 = np.array([3.1, 1.7, 1.1])

    print("---Первая СЛАУ---")
    sol_1 = Solution(a_1, b_1)
    sol_1.display_u_matrix()
    sol_1.display_solution()

    a_2 = np.array([
        [4.12, 0.42, 1.34, 0.88],
        [0.42, 3.95, 1.87, 0.43],
        [1.34, 0.87, 3.2, 0.31],
        [0.88, 0.43, 0.31, 5.17]
    ])
    b_2 = np.array([11.17, 0.115, 9.909, 9.349])

    print("---Вторая СЛАУ---")
    sol_2 = Solution(a_2, b_2)
    sol_2.display_u_matrix()
    sol_2.display_solution()
