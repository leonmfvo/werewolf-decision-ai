"""検証スクリプト共通: 昼開始ランダム排除モデルの勝率表（浮動小数点）。"""


def float_table(N, M, rule):
    """浮動小数点の表 T[n][m]（昼開始の人狼勝率、0 ≤ n ≤ N, 0 ≤ m ≤ M）。"""
    win = (lambda n, m: n >= m) if rule == "A" else (lambda n, m: n > m)
    T = [[0.0] * (M + 1)]
    for n in range(1, N + 1):
        row = [0.0] * (M + 1)
        prev = T[n - 1]
        for m in range(M + 1):
            if win(n, m):
                row[m] = 1.0
                continue
            p = n / (n + m)
            # 人狼が処刑された → (n-1, m) で夜
            if n == 1:
                a = 0.0
            elif win(n - 1, m) or win(n - 1, m - 1):
                a = 1.0
            else:
                a = prev[m - 1]
            # 村人が処刑された → (n, m-1) で夜
            b = 1.0 if (win(n, m - 1) or win(n, m - 2)) else row[m - 2]
            row[m] = p * a + (1 - p) * b
        T.append(row)
    return T
