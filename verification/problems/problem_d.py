"""人狼×数学 問題集の問題 D（mathematics/problems.qmd）の答えを数値で確かめる。

狼 1 人と村人 n 人の計 n+1 人。各人が自分以外の n 人から等確率・独立に 1 人へ投票する。
  X_n: 狼の得票数,  Y_n: 誰からも投票されなかった人の人数
答え: (1) lim P(X_n = k) = e^{-1}/k!   (2) lim V(Y_n)/n = (e-2)/e²

(2) は解答例と同じ分解 V(Y_n) = (n+1)p_n(1-p_n) + (n+1)n(q_n - p_n²) を使って厳密に計算し、
さらに小さい n でモンテカルロ法と比べる。
  p_n = P(ある人が無得票) = (1 - 1/n)^n
  q_n = P(ある 2 人がともに無得票) = (1 - 2/n)^(n-1) (1 - 1/n)^2

Python 3.8 以降、標準ライブラリのみ。
"""
from fractions import Fraction
from math import comb, e, exp, factorial
import random


def P_X(n, k):                      # X_n ~ 二項分布 B(n, 1/n)
    return comb(n, k) * Fraction(1, n) ** k * Fraction(n - 1, n) ** (n - k)


def var_Y(n):
    p = Fraction(n - 1, n) ** n
    q = Fraction(n - 2, n) ** (n - 1) * Fraction(n - 1, n) ** 2
    return (n + 1) * p * (1 - p) + (n + 1) * n * (q - p * p)


def simulate_var_Y(n, trials, rng):
    s = s2 = 0
    for _ in range(trials):
        got = [0] * (n + 1)
        for voter in range(n + 1):
            target = rng.randrange(n)
            got[target if target < voter else target + 1] += 1
        y = got.count(0)
        s += y
        s2 += y * y
    mean = s / trials
    return s2 / trials - mean * mean


if __name__ == "__main__":
    print("(1) P(X_n = k) と e^{-1}/k!")
    for k in range(5):
        row = "  ".join(f"n={n}: {float(P_X(n, k)):.6f}" for n in (10, 100, 1000))
        print(f"  k={k}: {row}   極限 {exp(-1) / factorial(k):.6f}")

    print("(2) V(Y_n)/n と (e-2)/e²")
    for n in (10, 100, 1000, 3000):
        print(f"  n={n:5d}: {float(var_Y(n)) / n:.6f}")
    print(f"  極限 (e-2)/e² = {(e - 2) / e ** 2:.6f}   独立と仮定した値 (e-1)/e² = {(e - 1) / e ** 2:.6f}")

    rng = random.Random(1)
    n, trials = 20, 200_000
    print(f"  モンテカルロ（n={n}, {trials} 回）: V(Y_n) ≈ {simulate_var_Y(n, trials, rng):.4f}"
          f"   厳密値 {float(var_Y(n)):.4f}")
