import numpy as np
from copy import deepcopy


def main_simplex_method(c, x, A, B):
    m, n = A.shape
    x_new = x.copy()
    assert (np.linalg.matrix_rank(A) == m)
    while True:
        AB = A[:, B]
        AB_inv = np.linalg.inv(AB)
        cB = c[B]
        u = cB @ AB_inv
        delta = u @ A - c
        if np.all(delta >= 0):
            return [x_new, B]
        j0 = np.where(delta < 0)[0][0]
        z = AB_inv @ A[:, j0]
        theta = np.array([x_new[B[i]] / z[i] if z[i] > 0 else np.inf for i in range(m)])
        theta0 = np.min(theta)
        if theta0 == np.inf:
            raise Exception("Целевая функция неограничена сверху на множестве допустимых планов!")
        k = np.argmin(theta)
        j_star = B[k]
        B[k] = j0
        for i in range(m):
            if i != k:
                x_new[B[i]] -= theta0 * z[i]
        x_new[j0] = theta0
        x_new[j_star] = 0


def initial_simplex_method(c, A, b):
    m, n = A.shape
    for i in range(len(b)):
        if b[i] < 0:
            b[i] *= -1
            A[i] *= -1
    c_tilde = np.concatenate((np.zeros(n), np.full((m,), -1)))
    A_tilde = np.concatenate((A, np.eye(m)), axis=1)
    x_tilde = np.concatenate((np.zeros(n), b))
    B = np.arange(n, n + m)
    x_tilde, B = main_simplex_method(c_tilde, x_tilde, A_tilde, B)
    if not np.all(x_tilde[n:n+m] == 0):
        print("Задача не совместна!")
    x = x_tilde[0:n]
    while True:
        if np.all((B >= 0) & (B < n - 1)):
            return x, B, A, b
        j_k = B[-1]
        i = j_k - n 
        k = B.shape[0] - 1  
        A_B_inv =  np.linalg.inv(A_tilde[:, B])
        l = np.array([A_B_inv @ A_tilde[:, j] for j in np.setdiff1d(np.arange(n), B)])
        if np.any(np.array([l_i[k] for l_i in l]) != 0):
            j = np.where([l_i[k] for l_i in l]) != 0
            B[k] = j
        else:
            A = np.delete(A, i, axis=0)
            A_tilde = np.delete(A_tilde, i, axis=0)
            b = np.delete(b, i, axis=0)
            B = np.delete(B, k, axis=0)


def solve(param_c, param_A, param_b):
    m, n = param_A.shape
    c = deepcopy(param_c)
    A = deepcopy(param_A)
    b = deepcopy(param_b)

    print(f"Vector c:\n{c}\n")
    print(f"Matrix A:\n{A}\n")
    print(f"Vector b:\n{b}\n")

    # ШАГ 1
    # ШАГ 2
    x_bar, B, _, _ = initial_simplex_method(c.copy(), A.copy(), b.copy())
    print(f"Vector x_bar:\n{x_bar}\n")
    print(f"Vector B:\n{B}\n")
    
    # ШАГ 3
    if all(i == int(i) for i in x_bar):
        print("========[Результат]========") 
        print(f"Оптимальный план:\nx: {x_bar}\nB: {B}\n")
    else:
        # ШАГ 4
        
        # ШАГ 5
        k = -1
        for index in range(0, len(x_bar), 1):
            if int(x_bar[index]) != x_bar[index]:
                k = index
                break
        x_bar_i = x_bar[k]
        print(f"Дробная компонента x_bar - x_bar[{k}]:\n{x_bar_i}\n")
        # ШАГ 6
        A_B = A[:, B]
        A_N = A[:, np.setdiff1d(np.arange(n), B)]
        print(f"Matrix A_B:\n{A_B}\n")
        print(f"Matrix A_N:\n{A_N}\n")
        # ШАГ 7
        x_B = x_bar[B]
        x_N = x_bar[np.setdiff1d(np.arange(n), B)]
        print(f"Vector x_B:\n{x_B}\n")
        print(f"Vector x_N:\n{x_N}\n")
        # ШАГ 8
        A_B_inv = np.linalg.inv(A_B)
        print(f"Matrix A_B_inv:\n{A_B_inv}\n")
        # ШАГ 9
        Q = A_B_inv @ A_N
        print(f"Matrix Q:\n{Q}\n")
        # ШАГ 10
        l = Q[k]
        print(f"Vector l:\n{l}\n")
        # ШАГ 11
        constraint = []
        constraint_c = np.zeros(n + 1)
        constraint_c[n] = -1
        # SUM[t, p]( {l_p} (x_N)_p ) - s = {x_bar_i}
        for p in range(len(l)):
            l_p_f = l[p] - np.floor(l[p])
            constraint_c[len(x_B) + p] = l_p_f
            constraint.append(f"{l_p_f}*X{len(x_B) + p + 1}")

        print("========[Результат]========")
        print(f"Отсекающее ограничение Гомори:\n{ " + ".join(constraint)} - s = {x_bar_i}\n")
        print(f"Вектор коэффициентов при переменных:\n{constraint_c}\n")
        print(f"Свободный член:\n{x_bar_i}\n")
            

if __name__ == "__main__":
    c = np.array([0, 1, 0, 0])
    A = np.array([
        [3, 2, 1, 0],
        [-3, 2, 0, 1]
    ])
    b = np.array([6, 0])
    solve(c, A, b)