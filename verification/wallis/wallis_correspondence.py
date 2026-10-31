"""論文「Wallis積と π/4 の出現」（mathematics/wallis-pi.qmd）の主張を再計算する。

  補題 3.1   kA 配役の初日村人処刑確率 p_k = (k-2)/(k-1)
  命題 3.3   R_n = Π_{j=2}^{n} p_{2j}/p_{2j+1} = 4^{2n-1} / (n C(2n,n)²)
  定理 3.4   R_n → π/4
  4.4 節     W_{2n} = 2(2n-1)!!/(2n)!!,  W_{2n+1} = (2n)!!/(2n+1)!!
  命題 4.4   S_n = Ω_{2n}/Ω_{2n+1} = W_{2n-1}/W_{2n} = 4^{2n-1} / (n C(2n,n)²)
  定理 5.1   R_n = S_n

人狼勝率 W_m は、閉形式を使わずにゲームそのもの
（人狼 1 人、生存者 m 人、昼に一様ランダム処刑、夜に村人 1 人を襲撃、人狼 ≥ 村人 で人狼勝利）
を厳密な有理数で計算して確かめる。

Python 3.8 以降、標準ライブラリのみ。
"""
from fractions import Fraction
from functools import lru_cache
from math import comb, pi, prod


def dfact(k):
    return prod(range(k, 0, -2)) if k > 0 else 1


@lru_cache(maxsize=None)
def W(m):
    """生存者 m 人（人狼 1・村人 m-1）で昼を迎えたときの人狼勝率。ゲームを直接計算する。"""
    villagers = m - 1
    if 1 >= villagers:                      # 人狼 ≥ 村人
        return Fraction(1)
    hit = Fraction(1, m)                    # 人狼が処刑される → 村勝利
    after = villagers - 1                   # 村人が処刑された
    if 1 >= after:
        return 1 - hit
    return (1 - hit) * W(m - 2)             # 夜の襲撃で村人がさらに 1 人減る


def p(k):                                   # kA 配役の初日村人処刑確率
    M = k - 1
    return Fraction(M - 1, M)


closed = lambda n: Fraction(4 ** (2 * n - 1), n * comb(2 * n, n) ** 2)

if __name__ == "__main__":
    NMAX = 400
    for n in range(1, NMAX):
        assert W(2 * n) == 2 * Fraction(dfact(2 * n - 1), dfact(2 * n)), n
        assert W(2 * n + 1) == Fraction(dfact(2 * n), dfact(2 * n + 1)), n
    print(f"4.4 節の閉形式 W_2n, W_2n+1: n < {NMAX} でゲームの直接計算と一致")

    R = Fraction(1)
    for n in range(2, NMAX):
        R *= p(2 * n) / p(2 * n + 1)
        S = W(2 * n - 1) / W(2 * n)          # Ω_{2n} / Ω_{2n+1}
        assert R == S == closed(n), n
    print(f"定理 5.1: R_n = S_n = 4^(2n-1)/(n C(2n,n)^2) が 2 ≤ n < {NMAX} で厳密に成立")
    print("  例: " + ", ".join(f"R_{n} = {closed(n)}" for n in range(2, 6)))

    print("定理 3.4: R_n → π/4")
    r = 1.0
    checkpoints = {10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6}
    for n in range(2, 10 ** 6 + 1):
        r *= (2 * n) * (2 * n - 2) / (2 * n - 1) ** 2
        if n in checkpoints:
            print(f"  n = {n:>7d}: R_n = {r:.8f}   R_n - π/4 = {r - pi / 4:+.2e}")
    print(f"  π/4 = {pi / 4:.8f}")
