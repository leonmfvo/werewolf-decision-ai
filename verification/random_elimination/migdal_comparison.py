"""§8 の比較表: Migdał (2010) の 1 次近似（式 (24)）と勝率五分の人狼数（式 (25)）を、
厳密値および §5.4 の極限の式と比べる。

strict 規約（人狼 > 村人 で人狼勝利）、昼開始、総人数 N・人狼 μ 人。
厳密値は Migdał の閉形式（式 (23)）を有理数で計算したもの
（critical_limits.py で漸化式と一致することを確認済み）。

Python 3.8 以降、標準ライブラリのみ。
"""
from math import pi, sqrt

from critical_limits import F_odd, F_strict_even, F_A_even, g, incl_excl


def bisect(f, target, a=0.01, b=3.0):
    for _ in range(100):
        mid = (a + b) / 2
        if f(mid) < target:
            a = mid
        else:
            b = mid
    return (a + b) / 2


if __name__ == "__main__":
    print(" N     μ   厳密値(strict)  Migdał 式(24)  極限の式")
    for mu in (10, 20, 30):
        for N in (mu * mu, mu * mu + 1):
            exact = float(incl_excl(mu, N, g))
            linear = mu * float(g(1, N))              # μ (N-1)!!/N!!
            c = mu / sqrt(N)
            limit = F_strict_even(c) if N % 2 == 0 else F_odd(c)
            print(f"{N:4d}  {mu:3d}   {exact:.5f}         {linear:.5f}        {limit:.5f}")

    print()
    print("勝率が 1/2 になる c = μ/√N")
    print(f"  N 偶数（strict）: 極限の式 {bisect(F_strict_even, 0.5):.4f}   Migdał 式(25) {0.5 * sqrt(pi / 2):.4f}")
    print(f"  N 奇数          : 極限の式 {bisect(F_odd, 0.5):.4f}   Migdał 式(25) {0.5 / sqrt(pi / 2):.4f}")
    print(f"  N 偶数（規則 A）: 極限の式 {bisect(F_A_even, 0.5):.4f}")
