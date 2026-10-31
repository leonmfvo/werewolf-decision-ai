"""研究ノート「小さな配役におけるCO戦略の厳密解析」に載せた数値をすべて再計算する。

各値は co_exact.solve で厳密な有理数として計算する。4 並列で数分かかる。
64 通りの進行すべての表は search_all.py（約 15 分）で作る。その出力は search_all_output.txt にある。
"""
from fractions import Fraction
from multiprocessing import Pool

from co_exact import CAST_4, CAST_7, Progression as P, solve

N2 = (None, None)

# (節, 説明, 配役, 進行, 人狼のモード, 騙りの固定, 記事に載せた値)
CASES = [
    # §3 村村狩狼
    ("3", "狩人1日目CO（人狼最善）", CAST_4, P(None, 1, "random", True, True), "best", None, "1/3"),
    ("3", "狩人2日目CO（人狼最善）", CAST_4, P(None, 2, "random", True, True), "best", None, "5/16"),
    ("3", "狩人潜伏（人狼最善）", CAST_4, P(None, None, "random", True, True), "best", None, "7/24"),
    ("3", "狩人1日目CO・人狼が対抗", CAST_4, P(None, 1, "random", True, True), "silent", {1: ("H",)}, "1/2"),
    ("3", "狩人潜伏・人狼が狩人を騙り本物が対抗", CAST_4, P(None, None, "random", True, True), "silent", {1: ("H",)}, "1"),
    ("3", "狩人潜伏・対抗なし・人狼が狩人を騙る", CAST_4, P(None, None, "random", True, False), "silent", {1: ("H",)}, "1"),
    # §4.1 対話ログのシミュレータ（固定ルールの人狼）の再現
    ("4.1", "能力なし", CAST_7, P(None, None, "none", False, False), "log", None, "8/35"),
    ("4.1", "霊・狩とも1日目CO", CAST_7, P(1, 1, "random", False, True), "log", None, "3/10"),
    ("4.1", "霊媒師のみ潜伏", CAST_7, P(None, 1, "random", False, True), "log", None, "11/45"),
    ("4.1", "狩人2日目CO・護衛なし", CAST_7, P(1, 2, "none", False, True), "log", None, "49/180"),
    ("4.1", "狩人2日目CO・霊媒師護衛", CAST_7, P(1, 2, "medium", False, True), "log", None, "49/180"),
    ("4.1", "狩人3日目CO・霊媒師護衛", CAST_7, P(1, 3, "medium", False, True), "log", None, "79/180"),
    ("4.1", "同・霊媒師への襲撃が1回失敗したら一様に戻る人狼", CAST_7, P(1, 3, "medium", False, True), "log_giveup", None, "239/720"),
    # §4.2 人狼の最善応答
    ("4.2", "霊・狩とも1日目CO・ランダム護衛・ローラー", CAST_7, P(1, 1, "random", True, True), "best", None, "3/10"),
    ("4.2", "同・ローラーなし", CAST_7, P(1, 1, "random", False, True), "best", None, "1/4"),
    ("4.2", "霊・狩とも1日目CO・霊媒師護衛・ローラー", CAST_7, P(1, 1, "medium", True, True), "best", None, "7/24"),
    ("4.2", "霊1日目CO・狩人潜伏・霊媒師護衛・ローラー", CAST_7, P(1, None, "medium", True, True), "best", None, "11/45"),
    ("4.2", "霊媒師潜伏・狩人1日目CO", CAST_7, P(None, 1, "random", False, True), "best", None, "17/75"),
    ("4.2", "霊1日目・狩2日目CO・ランダム護衛", CAST_7, P(1, 2, "random", False, True), "best", None, "29/150"),
    ("4.2", "霊1日目・狩2日目CO・霊媒師護衛", CAST_7, P(1, 2, "medium", False, True), "best", None, "1/5"),
    ("4.2", "霊1日目・狩3日目CO・霊媒師護衛", CAST_7, P(1, 3, "medium", False, True), "best", None, "5/32"),
    ("4.2", "同・騙りなしの人狼（襲撃だけ最善）", CAST_7, P(1, 3, "medium", False, True), "silent", None, "13/45"),
    # §4.3 仕組み
    ("4.3", "狩3日目: 人狼が3日目に狩人へ対抗", CAST_7, P(1, 3, "medium", False, True), "silent",
     {1: N2, 2: N2, 3: ("H", None)}, "5/32"),
    ("4.3", "狩3日目: 人狼が2日目に狩人を騙る", CAST_7, P(1, 3, "medium", False, True), "silent",
     {1: N2, 2: ("H", None)}, "5/9"),
    ("4.3", "1日目CO・霊媒師護衛: 人狼が霊媒師に対抗", CAST_7, P(1, 1, "medium", True, True), "best",
     {1: ("M", None)}, "7/24"),
    ("4.3", "1日目CO・ランダム護衛: 人狼が霊媒師に対抗", CAST_7, P(1, 1, "random", True, True), "best",
     {1: ("M", None)}, "19/60"),
    ("4.3", "1日目CO・ランダム護衛: 人狼が狩人に対抗", CAST_7, P(1, 1, "random", True, True), "best",
     {1: ("H", None)}, "7/20"),
]


def run(case):
    sec, name, cast, prog, mode, force, expected = case
    return case, solve(cast, prog, mode, force_claims=force)


if __name__ == "__main__":
    failed = 0
    with Pool() as pool:
        for case, v in pool.imap(run, CASES):
            sec, name, cast, prog, mode, force, expected = case
            ok = v == Fraction(expected)
            failed += not ok
            print(f"§{sec:3s} {cast.name:5s} {name:34s} {str(v):>8s} = {float(v):.4f}  {'OK' if ok else 'NG (記事: ' + expected + ')'}",
                  flush=True)
    if failed:
        raise SystemExit(f"{failed} 件が記事の値と一致しません")
    print("記事の値とすべて一致しました。")
