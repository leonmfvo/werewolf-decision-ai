# 自己対戦人狼AI（Werewolf AI）

研究ノート「[自己対戦による人狼AIの開発](https://leonmfvo.github.io/werewolf-decision-ai/research/self-play-ai.html)」で使ったコードです。
全プレイヤーを AI が演じる自己対戦シミュレータで、線形ソフトマックス方策を REINFORCE（価値関数をベースラインに使う）で学習します。

- Python 3、**標準ライブラリのみ**（追加のインストールは不要）
- コードは `src/` にあります。Claude との対話の中で作成し、手元で実行していた最終版（`werewolf_ai.zip`）をそのまま置いています。
- 学習済みの重み（9スタ、第 0〜1700 世代）は `weights/9std/` にあります。
- 使い方の詳細は [`src/README_使い方.md`](src/README_使い方.md) を見てください。

## 実行例

すべて `src/` の中で実行します。

```bash
cd ai/src
```

1 試合を表示する（9スタ配役）。

```bash
WEREWOLF_MODE=9std python3 main.py
```

重みを 0 から学習し、世代を保存して総当たりで比べる。

```bash
python3 generations.py reset --yes --mode 9std
```

```bash
python3 generations.py train --seconds 3600 --every 100 --mode 9std
```

```bash
python3 generations.py crossplay --games 500 --mode 9std --sort
```

リーグ学習（過去世代を練習相手にする）。

```bash
python3 generations.py league --seconds 3600 --every 100 --mode 9std --tag league
```

Windows の PowerShell では `WEREWOLF_MODE=9std` の代わりに `$env:WEREWOLF_MODE="9std"` を先に実行してください。
1 エポックは 1,000 試合で、手元の環境では 1 エポックに約 20 秒かかります。

## 配役（モード）

| モード | 配役 | 指定方法 |
|---|---|---|
| `absence`（既定） | 10a狩：10 人分の配役（狼 2・狂・占・霊・狩・村 4）から人狼以外の 1 枠を無作為に除いた 9 人。役欠けがある | 指定なし |
| `9std` | 9スタ：狼 2・狂・占・霊・狩・村 3（役欠けなし） | `--mode 9std`（`main.py` は環境変数 `WEREWOLF_MODE=9std`） |

重みと世代はモードごとに別のファイル・フォルダに保存されます（`trained_weights_9std.pkl`、`generations_9std/` など）。

## 主なファイル

| ファイル | 内容 | 記事の対応箇所 |
|---|---|---|
| `main.py` | ゲームエンジン。各陣営に、知ってよい情報だけを渡す | §3.1 |
| `config.py` | 配役とモードの設定 | §2.3 |
| `wolf_ai.py`、`village_ai.py`、`madman_ai.py`、`seer_ai.py`、`medium_ai.py`、`hunter_ai.py`、`villager_ai.py` | 役職ごとの AI | §3.1、§3.3、§3.4 |
| `policy_utils.py` | 線形ソフトマックス方策（`LinearPolicy`） | §3.2 |
| `value_function.py` | 状態に依存する価値関数（ベースライン） | §3.2、§5.2 |
| `board_split.py` | 盤面別学習（バケツ） | §3.5、§4.2 |
| `self_play.py` | 自己対戦と REINFORCE による重みの更新 | §3.2、§3.6 |
| `train_and_report.py` | 学習ループと、学習した方策の集計 | §3.6 |
| `generations.py` | 世代の保存、世代間の総当たり、リーグ学習 | §3.6、§4.3、§4.4 |
| `curriculum.py`、`strategy_search.py`、`policy_report.py` | カリキュラム学習、戦略の探索、方策の表示 | — |
| `persistence.py` | 重みの保存と読み込み | — |
| `log_export.py` | `main.py` の対戦ログを JSON に変換する | — |

## 学習済みの重み

`weights/9std/` に、9スタで重みを 0 から学習したときの世代を置いています（記事の §4.3 の総当たりで使った系統）。

| ファイル | 内容 |
|---|---|
| `gen_000100.pkl` 〜 `gen_001700.pkl` | 100 エポック（10 万試合）ごとの世代 |
| `gen_000200.pkl` | **第 200 世代**。総当たりで最も強かった世代で、公開用の重みとして選んだもの |
| `zero.pkl`、`gen_000000.pkl` | すべて 0 の重み（全員がランダムに行動する基準） |

第 200 世代で 1 試合を表示するには、`src/` で次のようにします。

```bash
cp ../weights/9std/gen_000200.pkl trained_weights_9std.pkl
```

```bash
WEREWOLF_MODE=9std python3 main.py
```

§4.3 の総当たりをやり直すには、世代を `src/generations_9std/` に置いてから `crossplay` を実行します（1 組 500 試合で、全体で数十分〜1 時間程度かかります）。

```bash
mkdir -p generations_9std && cp ../weights/9std/*.pkl generations_9std/
```

```bash
python3 generations.py crossplay --games 500 --mode 9std --sort
```

Windows の PowerShell では `cp` の代わりに `Copy-Item`、`mkdir -p` の代わりに `New-Item -ItemType Directory -Force` を使ってください。
乱数を固定していないので、総当たりの勝率は記事の値と試合数に応じた誤差（500 試合で ±2〜4 ポイント程度）の範囲でずれます。
この手順で再実行した結果は [`results/crossplay_9std_500games.txt`](results/crossplay_9std_500games.txt) にあります（総合勝率は記事の表と 0.5 ポイント以内で一致しました）。

## 公開していないもの

- **リーグ学習と独立系統の世代**（記事の §4.3 後半）。
- **7人村（霊狩2w）の段階のコード**。記事の §4.1 の結果は、このコードより前の版で得たものです。
- 対戦ログ。

`src/` で学習や総当たりを行うと生成されるファイル（`trained_weights*.pkl`、`generations*/`、`games.json`、`board_split_report.txt`）は `.gitignore` でコミットの対象外にしています。`weights/` の重みは書き換えられません。

## 注意

- 重みファイルは Python の pickle 形式です。pickle は読み込むとコードを実行できる形式なので、出所の確かなファイルだけを読み込んでください。`weights/9std/` の各ファイルは、辞書・リスト・数値・文字列だけでできていて、クラスや関数への参照を含まないことを確認しています。
