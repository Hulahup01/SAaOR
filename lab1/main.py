import numpy as np
import math
from copy import deepcopy


def dual_simplex_method(c, A, b, B) -> np.array:
    while True:
        len_b = len(B)
        A_b = np.zeros((len_b, len_b))
        for i in range(len_b):
            A_b[:, i] = A[:, B[i]]

        A_b_inv = np.linalg.inv(A_b)

        c_b = [c[i] for i in B]

        y = np.dot(c_b, A_b_inv)
        k = np.zeros(len(c))
        A_b_inv_b = np.dot(A_b_inv, b)
        for i in range(len(B)):
            k[B[i]] = A_b_inv_b[i]
        jk = -1
        is_optimal_plan = True

        for el in k:
            if el < 0:
                jk = el
                is_optimal_plan = False
                break

        if is_optimal_plan:
            return k

        k0 = list(k).index(jk)
        delta_y = A_b[B.index(k0)]

        not_B = list(set(range(len(c))) - set(B))
        mu = np.zeros(len(not_B))
        for j in range(len(mu)):
            mu[j] = np.dot(delta_y, A[:, not_B[j]])

        neg_mu = list()
        for elem in mu:
            if elem < 0:
                neg_mu.append(not_B[list(mu).index(elem)])
        if not neg_mu:
            return None

        sigmas = np.zeros(len(neg_mu))
        for j in range(len(sigmas)):
            i = neg_mu[j]
            sigmas[j] = (c[i] - np.dot(A[:, i], y)) / mu[j]

        sigma0 = min(sigmas)
        j0 = list(sigmas).index(sigma0)

        B[B.index(k0)] = not_B[j0]


def solve(param_c, param_A, param_b, param_d_l, param_d_r):
    m, n = param_A.shape
    c: np.array = deepcopy(param_c)
    A: np.array = deepcopy(param_A)
    b: np.array = deepcopy(param_b)
    d_l: np.array = deepcopy(param_d_l)
    d_r: np.array = deepcopy(param_d_r)

    print(f"Vector c:\n{c}\n")
    print(f"Matrix A:\n{A}\n")
    print(f"Vector b:\n{b}\n")
    print(f"Vector d-:\n{d_l}\n")
    print(f"Vector d+:\n{d_r}\n")

    # ШАГ 1
    for i in range(n):
        if c[i] > 0:
            c[i] = -c[i]  # Умножаем i-ую компоненту вектора c на -1
            A[:, i] = -A[:, i]  # Умножаем i-ый столбец матрицы A на -1
            d_l[i] = -d_l[i]  # Умножаем i-ую компоненту вектора d- на -1
            d_r[i] = -d_r[i]  # Умножаем i-ую компоненту вектора d+ на -1
            
            # Меняем местами d_l и d_r
            d_l[i], d_r[i] = d_r[i], d_l[i]

    print(f"Modified vector c:\n{c}\n")
    print(f"Modified matrix A:\n{A}\n")
    print(f"Vector b:\n{b}\n")
    print(f"Modified vector d-:\n{d_l}\n")
    print(f"Modified vector d+:\n{d_r}\n")

    # ШАГ 2
    alpha = 0
    c = np.concatenate((c, np.zeros(n + m)))
    A = np.hstack([np.vstack([A, np.eye(n)]), np.eye(m + n)])
    b = np.concatenate((b, d_r))
    d_l = np.concatenate((d_l, np.zeros(n + m)))

    # ШАГ 3
    x_star = []
    r = 0
    S = []
    delta = d_l
    S.append((c, A, b, d_l, delta, alpha))
    
    # ШАГ 4
    iter = 0
    while S:
        iter += 1
        print(f"=======ITER{iter}=======\n")

        s_c, s_A, s_b, s_d_l, s_delta, s_alpha = S.pop()
        print(f"Vector c:\n{s_c}\n")
        print(f"Matrix A:\n{s_A}\n")
        print(f"Vector b:\n{s_b}\n")
        print(f"Vector d-:\n{s_d_l}\n")
        print(f"Vector delta:\n{s_delta}\n")
        print(f"Alpha:\n{s_alpha}\n")

        s_alpha_ = s_alpha + np.dot(s_c, s_d_l)
        s_b_ = s_b - np.dot(s_A, s_d_l)
        s_B: list[int] = [n + i for i in range(n + m)]

        x_wave: np.array = dual_simplex_method(s_c, s_A, s_b_, s_B)

        if x_wave is None:
            print("no soulutions")
            continue
        else:
            print(f"x~: {x_wave}")

        # проверка на целочисленность
        is_integer = all(list(map(lambda x: x == int(x), x_wave)))

        if is_integer:
            x_hat = np.array(x_wave) + s_delta
    
            #  r < c * x_hat + alpha
            if len(x_star) == 0 or r < np.dot(s_c, x_hat)+ s_alpha_:
                r = np.dot(s_c, x_hat)+ s_alpha_
                x_star = deepcopy(x_hat)
            print(f"Int plan: x = {x_hat}, r = {r}")
        else:
            first_float = -1
            for index in range(0, len(x_wave), 1):
                if int(x_wave[index]) != x_wave[index]:
                     # выбираем дробную компоненту
                    first_float = index
                    break

        # [r < c * x~ + a`]
        if len(x_star) == 0 or r < math.floor(np.dot(s_c, x_wave) + s_alpha_):
            s_b__ = np.copy(s_b_)
            # заменой (m + i)-ой компоненты на ⌊x~i⌋
            s_b__[m + first_float] = math.floor(x_wave[first_float])
            # d− это (2n+m) 0vector. i-ой компоненты на ⌈x~i⌉.
            s_d_l_= np.zeros(2 * n + m)
            s_d_l_[first_float] = math.ceil(x_wave[first_float])
            s_delta_= s_delta + s_d_l_
          
            S.append((s_c, s_A, s_b__, np.zeros(2 * n + m), s_delta, s_alpha_))
            S.append((s_c, s_A, s_b_, s_d_l_, s_delta_, s_alpha_))

    if len(x_star) == 0:
        raise ValueError("The task is not compatible!")

    for i in range(n):
        if param_c[i] >= 0:
            x_star[i] *= -1
    return x_star[:n]
 
if __name__ == "__main__":
    c = np.array([1, 1])
    A = np.array([
        [5, 9],
        [9, 5]
    ])
    b = np.array([63, 63])
    d_l = np.array([1, 1])
    d_r = np.array([6, 6])

    x = solve(c, A, b, d_l, d_r)

    print(f"The optimal plan:\n x: {x}\n")

