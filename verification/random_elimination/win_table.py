"""ランダム排除人狼モデルの検証コード（昼開始・ルールA）

状態: 昼の開始時点で 狼 n 人・村人 m 人。
  昼: 生存者 n+m 人から一様ランダムに 1 人を処刑する。
  夜: 狼が生存村人から 1 人を襲撃する（村人は区別できないので選び方は結果に影響しない）。
  勝敗: 狼が 0 人になった時点で村勝利。狼の数 ≥ 村人の数 になった時点で狼勝利（処刑直後・襲撃直後とも判定）。
W[n][m] = 狼側の勝率。
"""
from fractions import Fraction
import math


def wolf_win_table(N, M, exact=False):
    one = Fraction(1) if exact else 1.0
    zero = Fraction(0) if exact else 0.0
    W = [[zero] * (M + 1)]                      # n = 0: 狼全滅 → 村勝利
    for n in range(1, N + 1):
        row = [zero] * (M + 1)
        for m in range(M + 1):
            if n >= m:                          # 既に狼 ≥ 村
                row[m] = one
                continue
            p_wolf = Fraction(n, n + m) if exact else n / (n + m)
            # 狼が処刑された場合: (n-1, m) で夜へ
            if n - 1 == 0:
                after_wolf = zero
            elif n - 1 >= m:
                after_wolf = one
            else:                               # 夜の襲撃で村人 -1
                after_wolf = W[n - 1][m - 1]
            # 村人が処刑された場合: (n, m-1) で夜へ
            if n >= m - 1:
                after_vil = one
            else:
                after_vil = row[m - 2]          # 夜の襲撃で村人がさらに -1
            row[m] = p_wolf * after_wolf + (1 - p_wolf) * after_vil
        W.append(row)
    return W


if __name__ == "__main__":
    # 1. 1狼の閉形式と積の恒等式（厳密な有理数で確認）
    E = wolf_win_table(3, 60, exact=True)
    dfact = lambda k: math.prod(range(k, 0, -2)) if k > 0 else 1
    for m in range(1, 60):
        closed = Fraction(dfact(m), dfact(m + 1)) * (2 if m % 2 else 1)
        assert E[1][m] == closed
        if m >= 2:
            assert E[1][m - 1] * E[1][m] == Fraction(2, m + 1)
    print("W_1(m) の閉形式, W_1(m-1)W_1(m) = 2/(m+1): OK (m < 60)")

    # 2. 一般の狼数での偶奇比 W_n(2k)/W_n(2k+1)
    W = wolf_win_table(4, 200001)
    k = 100000
    for n in range(1, 5):
        print(f"n={n}: W_n(2k)/W_n(2k+1) = {W[n][2*k]/W[n][2*k+1]:.5f}  "
              f"(π/4={math.pi/4:.5f}, 4/π={4/math.pi:.5f})")

    # 3. 臨界スケーリング（狼 n・村 n² 付近）
    S = wolf_win_table(160, 160**2 + 2)
    print(" n   W_n(n^2-1) [n+m 奇]   W_n(n^2) [n+m 偶]")
    for n in (10, 20, 40, 80, 160):
        print(f"{n:3d}   {S[n][n*n-1]:.6f}             {S[n][n*n]:.6f}")
    ext = lambda a, b: 2 * b - a                # L + C/n を仮定した外挿 (n → 2n)
    print("外挿 (80→160):", round(ext(S[80][6399], S[160][25599]), 5),
          round(ext(S[80][6400], S[160][25600]), 5))
