"""研究ノート「ランダム排除人狼モデルの一般化」の [数値] 主張をまとめて再計算する。

対応する節（mathematics/random-elimination-general.qmd）:
  §3.1  1 狼の勝率 W_1(m) の値
  §4.2  挟み込み W_1(m-n+1) ≤ W_n(m) ≤ n W_1(m-n+1)
  §4.4  strict 規約での偶奇比の極限（π/2 と 2/π の交代）
  §4.5  w(N,2) = 2 w(N,1)（N 偶数）など
  §5.1  「人狼 1・村人 5」を 1 組として組数を増やしたときの勝率
  §5.2  ポテンシャル Φ = w² - v の 1 ラウンドあたりの期待変化
  §5.5  のこぎり状の振動、勝率を一定に保つ数列（0.646 と 0.720）
  §5.6  極限が π/4 になる c、v/n² → ∞, 0 のときの傾向
  §6    狩人を加えた変形 Q(m) と P(m) の関係
  §7    f(1,3) < f(1,4)、1 狼勝率の漸近形、順列モデル
  §9    再生時刻の期待値 E[τ] と n-1+3/(n+2) の比較

モデル（昼開始）: 昼の開始時点で人狼 n 人・村人 m 人。
  昼: 生存者から一様ランダムに 1 人を処刑する。
  夜: 人狼が村人を 1 人襲撃する。
  勝敗: 人狼 0 人で村勝利。規則 A は「人狼 ≥ 村人」、strict は「人狼 > 村人」で人狼勝利
        （処刑直後・襲撃直後とも判定）。
夜開始の流儀 f(w, v) は昼開始の W_w(v-1) に等しい。

Python 3.8 以降、標準ライブラリのみ。実行時間は 1 分程度。
"""
from fractions import Fraction
from functools import lru_cache
from math import comb, erf, erfc, exp, log, pi, sqrt
import random
import sys

from model import float_table

sys.setrecursionlimit(100000)


# ---------------------------------------------------------------- 共通の道具
def exact_solver(rule):
    """厳密な有理数で昼開始の人狼勝率 day(n, m) を返す関数を作る。"""
    win = (lambda n, m: n >= m) if rule == "A" else (lambda n, m: n > m)

    @lru_cache(maxsize=None)
    def day(n, m):
        if n == 0:
            return Fraction(0)
        if win(n, m):
            return Fraction(1)
        p = Fraction(n, n + m)
        return p * night(n - 1, m) + (1 - p) * night(n, m - 1)

    @lru_cache(maxsize=None)
    def night(n, m):
        if n == 0:
            return Fraction(0)
        if win(n, m):
            return Fraction(1)
        return day(n, m - 1)

    return day


def dfact(k):
    r = 1
    for j in range(k, 0, -2):
        r *= j
    return r


def F_odd(c):
    return 1 - exp(-c * c / 2) + sqrt(2 * pi) * c * erfc(c / sqrt(2)) / 2


def F_A_even(c):
    return (1 + c * c) * erf(c / sqrt(2)) + sqrt(2 / pi) * c * exp(-c * c / 2) - c * c


def bisect(f, target, a, b):
    for _ in range(100):
        mid = (a + b) / 2
        if (f(mid) - target) * (f(a) - target) > 0:
            a = mid
        else:
            b = mid
    return (a + b) / 2


A = exact_solver("A")
S = exact_solver("strict")


def section(title):
    print()
    print("■ " + title)


# ---------------------------------------------------------------- §3.1
section("§3.1 1 狼の勝率 W_1(m), m = 1..8")
vals = [A(1, m) for m in range(1, 9)]
print("  ", ", ".join(str(v) for v in vals))
assert vals == [Fraction(x) for x in ("1", "2/3", "3/4", "8/15", "5/8", "16/35", "35/64", "128/315")]

# ---------------------------------------------------------------- §4.2
section("§4.2 挟み込み W_1(m-n+1) ≤ W_n(m) ≤ n W_1(m-n+1)")
for n in range(2, 7):
    for m in range(n + 1, 61):
        lo = A(1, m - n + 1)
        assert lo <= A(n, m) <= n * lo, (n, m)
print("   2 ≤ n ≤ 6, n < m ≤ 60 で成立（厳密な有理数）")

# ---------------------------------------------------------------- §4.4
section("§4.4 strict 規約での偶奇比 W_n(2k)/W_n(2k+1)  (k = 10^5)")
k = 10 ** 5
T = float_table(4, 2 * k + 1, "strict")
for n in range(1, 5):
    print(f"   n={n}: {T[n][2 * k] / T[n][2 * k + 1]:.5f}   (π/2={pi / 2:.5f}, 2/π={2 / pi:.5f})")
del T

# ---------------------------------------------------------------- §4.5
section("§4.5 strict 規約、総人数 N・人狼 μ 人の勝率 w(N, μ)")
w = lambda N, mu: S(mu, N - mu)
for N in range(4, 51):
    if N % 2 == 0:
        assert w(N, 2) == 2 * w(N, 1), N
    else:
        assert w(N, 2) == 2 * w(N, 1) - Fraction(dfact(N - 2), dfact(N)), N
print("   N 偶数: w(N,2) = 2w(N,1)、N 奇数: w(N,2) = 2w(N,1) - (N-2)!!/N!!  （4 ≤ N ≤ 50 で厳密に成立）")

# ---------------------------------------------------------------- §5.1
section("§5.1 「人狼 1・村人 5」を k 組（夜開始 f(k, 5k) = W_k(5k-1)）")
ks = (1, 2, 3, 4, 6, 8, 10, 12)
print("   " + "  ".join(f"k={k}: {float(A(k, 5 * k - 1)) * 100:.1f}%" for k in ks))

# ---------------------------------------------------------------- §5.2
section("§5.2 Φ = w² - v の 1 ラウンド（夜→昼）の期待変化")
for wv in range(1, 11):
    for v in range(wv + 3, 80):
        q = Fraction(wv, wv + v - 1)                      # 夜の後、昼に人狼が処刑される確率
        dphi = q * ((wv - 1) ** 2 - (v - 1) - (wv * wv - v)) + (1 - q) * (wv * wv - (v - 2) - (wv * wv - v))
        assert dphi == Fraction(2 * (v - (wv * wv - wv + 1)), wv + v - 1)
print("   E[ΔΦ] = 2(v - (w² - w + 1)) / (w + v - 1)  （1 ≤ w ≤ 10 で厳密に成立）")

# ---------------------------------------------------------------- §5.5
section("§5.5 のこぎり状の振動: 人狼 1+k・村人 5+10k（夜開始）")
print("   " + " → ".join(f"{float(A(1 + k, 4 + 10 * k)) * 100:.1f}%" for k in range(5)))

section("§5.5 勝率を一定に保つ数列: 人狼 k・村人 2k²（夜開始 f(k, 2k²) = W_k(2k²-1)）")
K = 120
T = float_table(K, 2 * K * K, "A")
seq = {k: T[k][2 * k * k - 1] for k in range(1, K + 1)}
for k in (10, 11, 20, 21, 40, 41, 80, 81, 119, 120):
    N = k + 2 * k * k - 1
    print(f"   k={k:3d}  総人数(昼開始) N={N:6d} ({'奇' if N % 2 else '偶'})  勝率 {seq[k]:.5f}")
ext = lambda a, b: 2 * b - a                               # L + C/k を仮定した外挿 (k → 2k)
print(f"   外挿: N 奇数の系列 {ext(seq[60], seq[120]):.4f}  (F_奇(1/√2) = {F_odd(1 / sqrt(2)):.4f})")
print(f"         N 偶数の系列 {ext(seq[59], seq[119]):.4f}  (F_偶^A(1/√2) = {F_A_even(1 / sqrt(2)):.4f})")

# ---------------------------------------------------------------- §5.6
section("§5.6 極限が π/4 になる c")
print(f"   N 奇数: c = {bisect(F_odd, pi / 4, 0.5, 1.5):.3f}   N 偶数・規則 A: c = {bisect(F_A_even, pi / 4, 0.5, 1.5):.3f}")

section("§5.6 v/n² → ∞ と v/n² → 0 の傾向（昼開始 W_n(v)）")
for n in (5, 10, 20, 40):
    v1 = int(n * n * log(n))
    v1 += (v1 + n) % 2                                     # 総人数を偶数にそろえる
    v2 = int(n ** 1.5)
    v2 += (v2 + n) % 2
    print(f"   n={n:2d}: v=n² ln n → {T[n][v1]:.4f}    v=n^1.5 → {T[n][v2]:.4f}")
del T

# ---------------------------------------------------------------- §6
section("§6 狩人を加えた変形（村勝率）")
# 狩人は処刑も襲撃もされない。昼は人狼と村人の中から 1 人を処刑、
# 夜は人狼が村人を 1 人襲撃し、狩人は自分以外（村人と人狼）から一様に 1 人を護衛する。
# 人狼勝利は「村人 ≤ 1」（人狼 ≥ 村人）、村勝利は人狼の処刑。


@lru_cache(maxsize=None)
def Q_day(v):                      # 昼の開始時点で村人 v 人（＋人狼 1、狩人 1）
    if v <= 1:
        return Fraction(0)
    p = Fraction(1, v + 1)         # 人狼が処刑される
    return p + (1 - p) * Q_night(v - 1)


@lru_cache(maxsize=None)
def Q_night(u):                    # 夜の開始時点で村人 u 人
    if u <= 1:
        return Fraction(0)
    g = Fraction(1, u + 1)         # 護衛成功（誰も死なない）
    return g * Q_day(u) + (1 - g) * Q_day(u - 1)


P = lambda m: 1 - A(1, m)          # 狩人なしの村勝率
for m in range(2, 9):
    target = P(m - 1) if m % 2 else P(m)
    assert Q_day(m) == target, m
    print(f"   m={m}: Q(m) = {Q_day(m)}  =  P({m - 1 if m % 2 else m})")

# ---------------------------------------------------------------- §7
section("§7 単調性の反例と 1 狼勝率の漸近形")
f = lambda wv, v: A(wv, v - 1)     # 夜開始
print(f"   f(1,3) = {f(1, 3)} < f(1,4) = {f(1, 4)}")
assert f(1, 3) < f(1, 4)
m = 2 * 10 ** 5 + 1
W1 = 2 * exp(sum(log(j) for j in range(m, 0, -2)) - sum(log(j) for j in range(m + 1, 0, -2)))
print(f"   m={m}: W_1(m) / (2√(2/(πm))) = {W1 / (2 * sqrt(2 / (pi * m))):.6f}")

section("§7 順列モデルと二重順列モデル（モンテカルロ、20 万回）")


def one_permutation(n, m, rng):
    line = [1] * n + [0] * m        # 1 = 人狼
    rng.shuffle(line)
    while True:
        line.pop(0)                 # 昼: 左端の人を処刑
        wv, v = sum(line), len(line) - sum(line)
        if wv == 0:
            return 0
        if wv >= v:
            return 1
        line.remove(0)              # 夜: 左端の村人を襲撃
        v -= 1
        if wv >= v:
            return 1


def two_permutations(n, m, rng):
    people = list(range(n + m))     # 0..n-1 が人狼
    day_order = people[:]
    rng.shuffle(day_order)
    night_order = list(range(n, n + m))
    rng.shuffle(night_order)
    alive = set(people)
    while True:
        alive.discard(next(x for x in day_order if x in alive))
        wv = sum(1 for x in alive if x < n)
        v = len(alive) - wv
        if wv == 0:
            return 0
        if wv >= v:
            return 1
        alive.discard(next(x for x in night_order if x in alive))
        if wv >= v - 1:
            return 1


rng = random.Random(1)
trials = 200_000
for n, m in ((2, 7), (2, 9)):
    exact = float(A(n, m))
    p1 = sum(one_permutation(n, m, rng) for _ in range(trials)) / trials
    p2 = sum(two_permutations(n, m, rng) for _ in range(trials)) / trials
    se = sqrt(exact * (1 - exact) / trials)
    print(f"   人狼 {n}・村人 {m}: 厳密値 {exact:.4f}  二重順列 {p2:.4f}  順列 {p1:.4f}  (標準誤差 {se:.4f})")

# ---------------------------------------------------------------- §9
section("§9 再生時刻の期待値 E[τ]（夜開始 (n, n²) から最初に人狼が処刑されるまでのラウンド数）")
# P(τ > j) = Π_{k<j} (n² - 2k - 1) / (n + n² - 2k - 1)。ゲームの終了は考えない。
for n in (4, 6, 8, 10, 12, 16, 20, 30):
    surv, E, k = Fraction(1), Fraction(0), 0
    while True:
        E += surv
        num = n * n - 2 * k - 1
        if num <= 0:
            break
        surv *= Fraction(num, n + num)
        k += 1
    diff = E - (n - 1 + Fraction(3, n + 2))
    print(f"   n={n:2d}: E[τ] = {float(E):.10f}   E[τ] - (n-1+3/(n+2)) = {float(diff):.3e}")

print()
print("すべての検証が完了しました。")
