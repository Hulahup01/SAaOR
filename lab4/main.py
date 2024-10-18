import numpy as np


def knapsack_problem(V, C, B):
    print(f"Вместимость B:\n{B}\n")
    print(f"Веса V:\n{V}\n")
    print(f"Стоимости C:\n{C}\n")

    n = len(V)
    # Прямой ход: заполнение таблицы OPT
    OPT = np.zeros((n + 1, B + 1))
    for k in range(1, n + 1):
        for b in range(B + 1):
            if V[k - 1] <= b:
                OPT[k][b] = max(OPT[k - 1][b], OPT[k - 1][b - V[k - 1]] + C[k - 1])
            else:
                OPT[k][b] = OPT[k - 1][b]

    # Обратный ход: восстановление выбранных предметов
    selected_items = []
    for k in range(n, 0, -1):
        if OPT[k][b] != OPT[k - 1][b]:  
            selected_items.append(k)  
            b -= V[k - 1]  

    return OPT[n, B], OPT, selected_items[::-1] 


if __name__ == '__main__':
    V = np.array([2, 4, 1, 2])
    C = np.array([7, 2, 5, 1])
    B = 6
    
    result, OPT, selected_items = knapsack_problem(V, C, B)
    print("========[Результаты]========") 
    print(f"OPT:\n{OPT}\n")
    print(f"Максимальная ценность:\n{result}\n")
    print(f"Выбранные предметы:\n{selected_items}\n")
