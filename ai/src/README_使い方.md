# 人狼AI 世代学習の使い方

すべて、このフォルダで実行します（Python 3、追加ライブラリ不要）。

1. リセット（現在の重みを generations/legacy_*.pkl に退避して 0 から開始）
   python3 generations.py reset --yes
2. 学習（時間で区切れます。再実行すると続きから。1000エポックごとに世代を保存）
   python3 generations.py train --seconds 3600 --every 1000
   - 学習中は1エポックごとに進捗を表示し、5分ごと・終了時に自動保存します。
3. 世代同士の総当たり（村側=世代A、狼側=世代B）
   python3 generations.py crossplay --games 500
   python3 generations.py report --games 500   # 表だけ再表示

- generations/zero.pkl はランダム基準、legacy_*.pkl は旧重み。どちらも総当たりに自動参加します。
- 世代が多いときは --stride 2 で間引けます。

## 9スタ配役（役欠けなし）で学習する場合
すべてのコマンドに `--mode 9std` を付けます（重みは trained_weights_9std.pkl、世代は generations_9std/ に別保存され、
通常（役欠けあり）の学習とは混ざりません）。
  python3 generations.py reset --yes --mode 9std
  python3 generations.py train --seconds 3600 --every 1000 --mode 9std
  python3 generations.py crossplay --games 500 --mode 9std
他方のモードの重みを読もうとすると、拒否してメッセージを出します。

## 別設定の学習を並行して残す / 学習率を変える
- `--tag 名前` を付けると、重みと世代フォルダが別になります（例: trained_weights_9std_lr003.pkl, generations_9std_lr003/）。
- `train --lr 0.003` で学習率を変えられます（既定 0.03）。
  python3 generations.py reset --yes --mode 9std --tag lr003
  python3 generations.py train --seconds 21600 --every 100 --lr 0.003 --mode 9std --tag lr003
- 同じ --mode / --tag を、reset・train・crossplay のすべてに付けてください。

## リーグ学習（過去世代を練習相手にする）
自己対戦だけだと、村と狼がお互いの癖に合わせすぎて、村がランダムより弱くなることがあります（観測済み）。
リーグ学習では、毎エポック次のどれかを行います。
- 自己対戦（両側が学習）… 既定で30%
- 村が学習し、狼は過去世代で固定
- 狼が学習し、村は過去世代で固定
相手は「自分の過去世代 ＋ --pool で指定した旧フォルダの世代」から選ばれ、まだ負けている相手ほど選ばれやすくなります。
固定した側の重みは一切変わりません（Ctrl+C で途中停止しても元に戻してから保存します）。

例: 旧9スタ学習の世代（generations_9std/）を練習相手にして、ゼロから別系統（タグ league）を学習
  python3 generations.py reset --yes --mode 9std --tag league
  python3 generations.py league --seconds 3600 --every 100 --mode 9std --tag league
  （--tag を付けると、同じモードのタグ無しフォルダ generations_9std/ が自動で練習相手になります。変えるときは --pool フォルダ名、無効にするときは --pool none）
- 調整: --self-frac 0.3（自己対戦の割合）、--pfsp 1.0（負けている相手を優先する強さ。0で一様）、--lr、--games
- 学習中の表示例: epoch 120 (this run: 20)  [village learns vs o_g1400 wolves]  village win 63.0%
  （o_ が付くのは --pool 側の旧世代。village win は、どの行でも常に村側の勝率です。「village learns」の行なら学習側の勝率、「wolf learns」の行なら学習側の負け率にあたります）

世代の評価（旧世代も混ぜて総当たり。旧フォルダで終わっている組み合わせは再利用されます）
  python3 generations.py crossplay --games 500 --mode 9std --tag league --pool generations_9std --sort
- 世代が20を超えると行列は幅が広すぎるので省略されます（--matrix で表示）。
- 強さの表の worst(v)/worst(w) は、相手の世代1つに対する最悪の成績です（平均だけでなく崩れにくさも見られます）。
- 同じ --pool を、crossplay と report の両方に付けてください。
