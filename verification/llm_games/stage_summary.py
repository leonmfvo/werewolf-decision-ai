"""研究ノート「LLMで人狼プレイヤーを作る」シリーズの対局データ集計を再現する。

入力（同じディレクトリ）:
  games.csv       1 行 = 1 試合（55 試合）。試合番号は日時順の通し番号で、ログ ID・村名・プレイヤー名は含まない。
  utterances.csv  1 行 = 試合中の 1 発言（AI と、その試合に参加した人間プレイヤー）。本文は含まず、
                  文字数と「」・アンカー（>>）・疑問符・ｗ の有無だけを持つ。

段階（stage）:
  S1 ローカル LLM / S2 API 移行初期 / S3 人格・考える係と話す係の分業 /
  S4 統計による文体の模倣（多層化） / S5 素の LLM

使い方:  python3 stage_summary.py
"""
import csv
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
STAGES = ["S1", "S2", "S3", "S4", "S5"]


def load(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wilson(k, n, z=1.96):
    """二項比率の 95% Wilson 区間。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def frac(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}) [{lo:.0%}, {hi:.0%}]" if n else "—"


def share(rows, key):
    return sum(int(r[key]) for r in rows) / len(rows)


def main():
    games = load("games.csv")
    utts = load("utterances.csv")
    assert len(games) == 55, len(games)
    ai_utts = [u for u in utts if u["speaker"] == "AI"]
    assert len(ai_utts) == sum(int(g["ai_posts"]) for g in games) == 921

    print("== 全体 ==")
    wins = sum(int(g["ai_win"]) for g in games)
    print("試合数", len(games), " AI の勝ち", frac(wins, len(games)))
    for side, label in (("village", "村側"), ("werewolf", "人狼側")):
        gs = [g for g in games if g["ai_side"] == side]
        print(f"  {label}", frac(sum(int(g["ai_win"]) for g in gs), len(gs)))
    four = [g for g in games if g["jobset"] == "wolf1 villager3" and g["missing_roles"] == "yes"]
    for side, label in (("village", "村側"), ("werewolf", "人狼側")):
        gs = [g for g in four if g["ai_side"] == side]
        print(f"  4A（狼1・村3）の{label}", frac(sum(int(g["ai_win"]) for g in gs), len(gs)))
    # 4A では昼 1 回の吊りで決着する。3 人が無作為に投票すれば村勝率は 1/3。
    print("  参考: 4A で全員が無作為に投票した場合の村勝率 = 1/3")

    print("\n== 段階別（表） ==")
    header = ["段階", "試合", "勝ち", "村側", "人狼側", "投票で人狼を指名", "発言/試合",
              "字数中央値", "100字超", "「」", ">>", "人間の字数中央値"]
    print(" | ".join(header))
    for s in STAGES:
        gs = [g for g in games if g["stage"] == s]
        vil = [g for g in gs if g["ai_side"] == "village"]
        wolf = [g for g in gs if g["ai_side"] == "werewolf"]
        votes = [g for g in vil if g["players"] == "4" and g["ai_vote_hit"] != ""]
        a = [u for u in ai_utts if u["stage"] == s]
        h = [u for u in utts if u["stage"] == s and u["speaker"] == "human"]
        lens = [int(u["length"]) for u in a]
        row = [
            s, str(len(gs)),
            f"{sum(int(g['ai_win']) for g in gs)}/{len(gs)}",
            f"{sum(int(g['ai_win']) for g in vil)}/{len(vil)}",
            f"{sum(int(g['ai_win']) for g in wolf)}/{len(wolf)}",
            f"{sum(int(g['ai_vote_hit']) for g in votes)}/{len(votes)}",
            f"{len(a) / len(gs):.1f}",
            f"{statistics.median(lens):.0f}",
            f"{sum(1 for x in lens if x > 100) / len(lens):.0%}",
            f"{share(a, 'quote'):.0%}",
            f"{share(a, 'anchor'):.0%}",
            f"{statistics.median(int(u['length']) for u in h):.0f}",
        ]
        print(" | ".join(row))

    print("\n== 段階別の勝率と 95% 区間 ==")
    for s in STAGES:
        gs = [g for g in games if g["stage"] == s]
        vil = [g for g in gs if g["ai_side"] == "village"]
        print(s, "全体", frac(sum(int(g["ai_win"]) for g in gs), len(gs)),
              " 村側", frac(sum(int(g["ai_win"]) for g in vil), len(vil)))

    print("\n== S5 の村側 7 試合の内訳 ==")
    for g in games:
        if g["stage"] == "S5" and g["ai_side"] == "village":
            print(f"  試合{g['game']:>2}  {g['jobset']}（役欠け{g['missing_roles']}）  勝利: {g['winner']}"
                  f"  AIの投票が人狼: {g['ai_vote_hit']}")

    print("\n== 村側・4人村での負け方 ==")
    for s in STAGES:
        v = [g for g in games if g["stage"] == s and g["ai_side"] == "village" and g["players"] == "4"]
        lynched = sum(int(g["ai_died"]) for g in v)
        lost_after_hit = sum(1 for g in v if g["ai_vote_hit"] == "1" and g["ai_win"] == "0")
        print(f"{s}: {len(v)} 試合中、AI が処刑された {lynched}、AI は人狼に投票したが負けた {lost_after_hit}")

    print("\n== 開発者の試験村（test_room=yes）とそれ以外 ==")
    for s in STAGES:
        for t, label in (("yes", "試験村"), ("no", "それ以外")):
            gs = [g for g in games if g["stage"] == s and g["test_room"] == t]
            if not gs:
                continue
            vil = [g for g in gs if g["ai_side"] == "village"]
            print(f"{s} {label}: {len(gs)} 試合  勝ち {sum(int(g['ai_win']) for g in gs)}/{len(gs)}"
                  f"  村側 {sum(int(g['ai_win']) for g in vil)}/{len(vil)}")

    print("\n== AI と人間の発言の比較（全試合） ==")
    hum = [u for u in utts if u["speaker"] == "human"]
    for label, rows in (("AI", ai_utts), ("人間", hum)):
        lens = [int(u["length"]) for u in rows]
        print(f"{label}: 発言 {len(rows)}  字数 平均 {statistics.mean(lens):.1f} 中央値 {statistics.median(lens):.0f}"
              f"  100字超 {sum(1 for x in lens if x > 100) / len(lens):.0%}"
              f"  「」 {share(rows, 'quote'):.0%}  >> {share(rows, 'anchor'):.0%}"
              f"  疑問符 {share(rows, 'question'):.0%}  ｗ {share(rows, 'laugh'):.0%}")

    # 村側の投票（4 人村）：AI と人間
    v = [g for g in games if g["players"] == "4" and g["ai_side"] == "village" and g["ai_vote_hit"] != ""]
    print("\nAI（村側・4人村）の投票が人狼に当たった割合", frac(sum(int(g["ai_vote_hit"]) for g in v), len(v)),
          " ※無作為なら 1/2")
    hv = sum(int(g["human_village_votes"]) for g in games if g["human_village_votes"] != "")
    hh = sum(int(g["human_village_vote_hits"]) for g in games if g["human_village_vote_hits"] != "")
    print("人間（村側・同じ 4人村）の投票が人狼に当たった割合", frac(hh, hv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
