"""§5.4 の極限値の式を、動的計画法（厳密な漸化式の数値計算）の外挿と比べる。

総人数 N・人狼 μ 人（昼開始）で μ = c√N を保って N → ∞ とする。
人狼 n = 100, 200 で N ≈ n²/c² の盤面を計算し、L + C/n を仮定して外挿する。
規則 A と strict のそれぞれで、総人数 N の偶奇ごとに比べる。

Python 3.8 以降、標準ライブラリのみ。実行時間は数分程度（純 Python で約 6,000 万マス）。
"""
from math import erf, erfc, exp, pi, sqrt

from model import float_table


def F_odd(c):                       # N 奇数（規則 A・strict 共通）
    return 1 - exp(-c * c / 2) + sqrt(2 * pi) * c * erfc(c / sqrt(2)) / 2


def F_strict_even(c):
    return erf(c / sqrt(2))


def F_A_even(c):
    return (1 + c * c) * erf(c / sqrt(2)) + sqrt(2 / pi) * c * exp(-c * c / 2) - c * c


def extrapolate(T, c, parity, n1=100, n2=200):
    vals = []
    for n in (n1, n2):
        N = round(n * n / (c * c))
        if N % 2 != parity:
            N += 1
        vals.append(T[n][N - n])
    return 2 * vals[1] - vals[0]


if __name__ == "__main__":
    cs = (0.5, 1.0, 1.5)
    M = int(200 * 200 / min(cs) ** 2) + 4
    TA = float_table(200, M, "A")
    TS = float_table(200, M, "strict")
    print("  c   | 偶数・規則A  DP / 式      | 偶数・strict  DP / 式     | 奇数  DP(A) / DP(strict) / 式")
    for c in cs:
        print(f" {c:.1f}  | {extrapolate(TA, c, 0):.5f} / {F_A_even(c):.5f}   "
              f"| {extrapolate(TS, c, 0):.5f} / {F_strict_even(c):.5f}   "
              f"| {extrapolate(TA, c, 1):.5f} / {extrapolate(TS, c, 1):.5f} / {F_odd(c):.5f}")
