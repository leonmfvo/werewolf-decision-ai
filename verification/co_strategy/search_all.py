"""すべての進行について、人狼の最善応答に対する村の勝率を厳密に計算し、表にする。

使い方:  python3 search_all.py [4] [7]      （引数なしなら両方。7 人は 4 並列で 20〜30 分程度）
"""
import sys
import time
from multiprocessing import Pool

from co_exact import CAST_4, CAST_7, all_progressions, solve

CASTS = {"4": CAST_4, "7": CAST_7}


def evaluate(args):
    key, p = args
    cast = CASTS[key]
    return p, solve(cast, p, "best"), solve(cast, p, "silent")


if __name__ == "__main__":
    keys = sys.argv[1:] or ["4", "7"]
    for key in keys:
        cast = CASTS[key]
        t0 = time.time()
        jobs = [(key, p) for p in all_progressions(cast)]
        with Pool() as pool:
            rows = []
            for p, best, silent in pool.imap_unordered(evaluate, jobs):
                rows.append((best, silent, p))
                print(f"  ... {p.label()}: {float(best):.4f}", file=sys.stderr, flush=True)
        rows.sort(key=lambda r: (-r[0], -r[1], r[2].label()))
        print(f"== {cast.name}（{len(rows)} 通りの進行、{time.time() - t0:.0f} 秒）")
        print("順位  村勝率（人狼が最善）       騙りなしの人狼に対して      進行")
        for i, (b, s, p) in enumerate(rows, 1):
            print(f"{i:3d}   {float(b):.4f} ({str(b):>11s})   {float(s):.4f} ({str(s):>11s})   {p.label()}")
        print()
