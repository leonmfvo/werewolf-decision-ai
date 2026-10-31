"""臨界スケーリング（人狼 μ = c√N）の極限値の検証（昼開始）

rule="A"      : 人狼の数 ≥ 村人の数 で人狼勝利（本ノートの標準）
rule="strict" : 人狼の数 > 村人の数 で人狼勝利（Migdał の規約。村人 = 0 ルールと同じ結果）
"""
from fractions import Fraction
from functools import lru_cache
from math import comb, erf, erfc, exp, pi, sqrt
import sys

sys.setrecursionlimit(100000)


def solver(rule):
    win = (lambda n, m: n >= m) if rule == "A" else (lambda n, m: n > m)

    @lru_cache(maxsize=None)
    def day(n, m):                      # 昼の開始時点の人狼勝率（厳密な有理数）
        if n == 0:
            return Fraction(0)
        if win(n, m):
            return Fraction(1)
        p = Fraction(n, n + m)
        return p * night(n - 1, m) + (1 - p) * night(n, m - 1)

    @lru_cache(maxsize=None)
    def night(n, m):                    # 夜の開始時点（襲撃で村人 -1）
        if n == 0:
            return Fraction(0)
        if win(n, m):
            return Fraction(1)
        return day(n, m - 1)

    return day


def dfact(k):                           # 負の二重階乗を含む (Migdał の規約)
    if k >= -1:                         # (-1)!! = 0!! = 1
        r = 1
        for j in range(k, 0, -2):
            r *= j
        return Fraction(r)
    if k % 2 == 0:
        return None                     # 1/(-2j)!! = 0
    j = (-k - 1) // 2                   # (-2j-1)!! = (-1)^j / (2j-1)!!
    return Fraction((-1) ** j) / dfact(2 * j - 1)


def g(i, N):                            # Migdał の g_i(N)
    d = dfact((N % 2) - i)
    return Fraction(0) if d is None else dfact(N - i) / (dfact(N) * d)


def coef_A(i, N):                       # 規則 A・N 偶数の係数（本ノートで見つけた形）
    if i % 2 == 1:
        j = (i - 1) // 2
        return Fraction(-2, 2 * j - 1) * g(i, N)
    return Fraction(2, N) if i == 2 else Fraction(0)


def incl_excl(mu, N, coef):
    return sum((-1) ** (i + 1) * comb(mu, i) * coef(i, N) for i in range(1, mu + 1))


def F_odd(c):                           # N 奇数（両規則で共通）
    Phi_c = 1 - erfc(c / sqrt(2)) / 2
    return 1 - exp(-c * c / 2) + sqrt(2 * pi) * c * (1 - Phi_c)


def F_strict_even(c):
    return erf(c / sqrt(2))


def F_A_even(c):
    return (1 + c * c) * erf(c / sqrt(2)) + sqrt(2 / pi) * c * exp(-c * c / 2) - c * c


if __name__ == "__main__":
    A, S = solver("A"), solver("strict")

    # 1. 有限の N での厳密な恒等式
    for N in range(4, 41):
        for mu in range(1, min(N // 2, 8) + 1):
            m = N - mu
            assert S(mu, m) == incl_excl(mu, N, g)                 # Migdał の閉形式
            if N % 2 == 1:
                assert A(mu, m) == S(mu, m)                        # N 奇数: 規則 A = strict
            else:
                assert A(mu, m) == incl_excl(mu, N, coef_A)        # N 偶数: 規則 A の展開
    print("Migdał の閉形式・規則 A の展開: N ≤ 40, μ ≤ 8 で厳密に一致")

    # 2. 極限値（c = 1）
    print(f"N 奇数            : {F_odd(1):.6f}")
    print(f"N 偶数・strict     : {F_strict_even(1):.6f}")
    print(f"N 偶数・規則 A     : {F_A_even(1):.6f}")
    print(f"村 n²・狼 n の村勝率（規則 A）: {1 - F_A_even(1):.6f}")
    print(f"村 n²・狼 n の村勝率（村人 = 0 ルール）: {1 - F_strict_even(1):.6f}")
