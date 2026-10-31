# 検証コード

サイトの記事に載せた数式・数値を再現するためのスクリプトです。
すべて Python 3.8 以降の **標準ライブラリだけ** で動きます（追加のインストールは不要です）。

```bash
python3 verification/run_all.py
```

で全スクリプトを実行します（数分かかります）。個別に実行する場合は、各スクリプトのあるディレクトリで `python3 <ファイル名>` としてください。
GitHub Actions（`.github/workflows/verify.yml`）でも、`verification/` を変更したときに同じものを実行しています。

## 記事との対応

| スクリプト | 記事 | 再現する内容 |
|---|---|---|
| [`wallis/wallis_correspondence.py`](wallis/wallis_correspondence.py) | [Wallis積と π/4 の出現](https://leonmfvo.github.io/werewolf-decision-ai/mathematics/wallis-pi.html) | 人狼勝率の閉形式（4.4 節）をゲームの直接計算で確認、$R_n=S_n=4^{2n-1}/(n\binom{2n}{n}^2)$（定理 5.1）、$R_n\to\pi/4$（定理 3.4） |
| [`random_elimination/win_table.py`](random_elimination/win_table.py) | [ランダム排除人狼モデルの一般化](https://leonmfvo.github.io/werewolf-decision-ai/mathematics/random-elimination-general.html) 付録 A | 1 狼の閉形式と $W_1(m-1)W_1(m)=2/(m+1)$（§3）、偶奇比の表（§4.1）、臨界配役の勝率表と外挿値（§5.3） |
| [`random_elimination/critical_limits.py`](random_elimination/critical_limits.py) | 同 付録 B | Migdał の閉形式と漸化式の一致、規則 A の包除係数（$N\le40$, $\mu\le8$）、極限値 0.791159 / 0.682689 / 0.849320 / 0.150680 / 0.317311（§5.4） |
| [`random_elimination/limits_vs_dp.py`](random_elimination/limits_vs_dp.py) | 同 §5.4 | $c=0.5,1,1.5$ で極限の式と動的計画法の外挿値を比較 |
| [`random_elimination/numerical_claims.py`](random_elimination/numerical_claims.py) | 同 §3–§7, §9 | 本文中の **[数値]** の主張（挟み込み、strict 規約の偶奇比、$w(N,2)=2w(N,1)$、比率一定の表、のこぎり状の振動、0.646 / 0.720 の 2 値、$\pi/4$ になる $c$、狩人の恒等式、順列モデル、$\mathbb{E}[\tau]$） |
| [`random_elimination/migdal_comparison.py`](random_elimination/migdal_comparison.py) | 同 §8 | Migdał (2010) の 1 次近似・勝率五分の人狼数と、厳密値・極限の式の比較表 |
| [`random_elimination/model.py`](random_elimination/model.py) | — | 上のスクリプトが共通に使う勝率表（浮動小数点の動的計画法） |
| [`problems/problem_d.py`](problems/problem_d.py) | [人狼×数学 問題集](https://leonmfvo.github.io/werewolf-decision-ai/mathematics/problems.html) 問題 D | $P(X_n=k)\to e^{-1}/k!$、$V(Y_n)/n\to(e-2)/e^2$（厳密計算とモンテカルロ法） |

## モデルの約束

- **昼開始**：昼の開始時点で人狼 $n$ 人・村人 $m$ 人。昼に生存者から一様ランダムに 1 人を処刑し、夜に人狼が村人を 1 人襲撃する。
- **規則 A**：人狼 $\ge$ 村人 になった時点で人狼勝利。**strict**：人狼 $>$ 村人 で人狼勝利（Migdał (2010) の規約）。どちらも人狼 0 人で村勝利。
- 夜開始の流儀 $f(w,v)$ は昼開始の $W_w(v-1)$ に等しい。

## 公開していないもの

- 研究ノート「[自己対戦による人狼AIの開発](https://leonmfvo.github.io/werewolf-decision-ai/research/self-play-ai.html)」の学習コード（`main.py`、`generations.py` など）と学習済みの重みは、まだこのリポジトリにありません。
