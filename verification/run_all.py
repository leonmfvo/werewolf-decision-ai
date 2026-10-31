"""verification/ 以下の検証スクリプトをすべて実行し、失敗があれば終了コード 1 で終わる。

使い方:  python3 verification/run_all.py
"""
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
SCRIPTS = [
    "wallis/wallis_correspondence.py",
    "random_elimination/win_table.py",
    "random_elimination/critical_limits.py",
    "random_elimination/numerical_claims.py",
    "random_elimination/migdal_comparison.py",
    "random_elimination/limits_vs_dp.py",
    "problems/problem_d.py",
    "llm_games/stage_summary.py",
]

failed = []
for rel in SCRIPTS:
    path = ROOT / rel
    print(f"===== {rel}", flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, path.name], cwd=path.parent)
    print(f"----- {rel}: {'OK' if result.returncode == 0 else 'FAILED'} ({time.time() - t0:.1f} 秒)\n", flush=True)
    if result.returncode != 0:
        failed.append(rel)

if failed:
    print("失敗:", ", ".join(failed))
    sys.exit(1)
print(f"{len(SCRIPTS)} 本すべて成功")
