"""
Standalone training program.

- Loads previously saved weights if they exist (continues training
  instead of restarting from zero).
- Trains for NUM_EPOCHS more epochs.
- Saves the result.
- Prints a full list of what the Village AI has learned for every
  representative board situation, sorted from "most suspicious"
  to "least suspicious".

Usage:
    python3 train_and_report.py

Run it again later and it picks up where it left off.
"""

import random
from collections import defaultdict

import main
import self_play
import wolf_ai
import village_ai
import medium_ai
import hunter_ai
import seer_ai
import villager_ai
import madman_ai
import value_function
import policy_utils
import board_split
import persistence


# NUM_EPOCHS -- sized for testing the per-Day-1-board-config split
# (board_split.py / enable_full_board_split()), not just ordinary
# training. Recommendation is based on measured Day-1 config
# frequencies (train_and_report.day1_config_report()): the rarest of
# the ORIGINAL 10 "common" buckets, (2,1,1), shows up in only ~1.3% of
# games -- at 1,000 epochs (1,000,000 games total, same overall
# budget as before) that bucket alone accumulates roughly 13,000
# samples, enough for a first meaningful per-bucket win-rate
# comparison against the shared baseline. This does NOT rescue the
# long tail enable_full_board_split() adds on top of that, though --
# several of those configs are below 0.1% of games (see
# day1_config_report's own printed breakdown), so even 1,000,000
# games leaves them with well under 1,000 samples each; that's an
# inherent cost of full splitting, not something a bigger NUM_EPOCHS
# alone fixes (see enable_full_board_split's docstring).
# At ~150 games/sec (this project's own measured throughput, 2026-09
# -- expect it to vary with your hardware), 1,000,000 games is
# roughly 1.8 hours. Safe to interrupt any time (Ctrl+C) and resume
# later -- persistence.py saves progress after every train() run, not
# just at the very end.
#
# GAMES_PER_EPOCH raised from 50 to 1000 (2026-09-30, per chat): the
# printed per-epoch win rate is a snapshot of ONLY that epoch's own
# games (see the loop below -- results/village_wins are recreated
# fresh every epoch, never accumulated across epochs), so its
# binomial sampling noise is set entirely by this one number. At 50,
# the 95% CI on a single print is roughly +-14 points -- easily wide
# enough on its own to produce something like an 8%->98% jump between
# two prints even if the TRUE win rate barely moved; a live 11,150-
# epoch log measured this session showed a full-run stdev around 17-21
# points with no clear sign of narrowing, i.e. mostly this noise floor,
# not a converging value. At 1,000 games the same 95% CI on a single
# print shrinks to roughly +-3 points, so a swing that size actually
# means something moved, not just an unlucky small sample. NUM_EPOCHS
# is lowered from 20,000 to 1,000 in the same edit to keep the total
# game budget (and wall-clock time) roughly unchanged -- this is a
# CHANGE IN HOW OFTEN A GRADIENT UPDATE HAPPENS too (each epoch is
# still exactly one REINFORCE update over its GAMES_PER_EPOCH games),
# so this run does 20x fewer, 20x-larger-batch updates than the
# previous 20,000-epoch/50-game-batch setting for the same total
# games -- a real change to training dynamics, not just to what gets
# printed, though the two are hard to fully separate without an A/B
# comparison at matched total games.
NUM_EPOCHS = 1000
GAMES_PER_EPOCH = 1000
LEARNING_RATE = 0.03
DECAY_RATE = 0.03  # how fast old, unsupported weights fade toward zero each epoch
PRINT_EVERY = 5  # every 5 epochs = every 5,000 games, ~33 sec at ~150 games/sec
CHECKPOINT_EVERY = 50  # epochs between atomic mid-run saves

# Switched on by default: a plain running-average baseline turned out
# to be nowhere near enough for the wolf vote's "protect my teammate"
# decision -- that signal is rare and easy to drown in noise, and a
# side-by-side comparison showed its learned weight oscillating
# between roughly -1.9 and +1.9 even after 1200 epochs (60,000
# games) with the plain baseline, never settling on a sign. Switching
# to the state-dependent value_function critic fixed this completely
# in the same comparison -- the weight stayed consistently negative
# (roughly -0.7 to -1.9) across the same length of training. Since
# it's a strict upgrade to credit assignment for every decision, not
# just this one, it's now the default for every side.
USE_VALUE_FUNCTION = True
VALUE_LEARNING_RATE = 0.02


# ============================================================
# Training
# ============================================================

def _extract_day1_config(result):
    """
    (Sy, Ps, Hn) for one run_game() result, read from the LAST history
    record that carries a non-None "day1_config" -- NOT the first.

    main.py's _current_day1_config() is only stable once day_phase(1)'s
    entire CO sequence has finished (its own docstring explains why: a
    call mid-sequence sees only a partial count). Every history record
    logged DURING Day 1 carries whatever the count happened to be at
    that moment, so grabbing the first match can return an early,
    incomplete value (e.g. (0,1,0) from a Hunter's own co_action
    decision, logged before the Seer or the wolves/Madman have even
    acted yet) -- a real bug hit and fixed once already this project
    (see the full-board-split test's first, silently-all-"other"-
    miscategorized run). Reading in reverse and taking the first hit
    always lands on the final, settled value instead.
    """
    for record in reversed(result["history"]):
        cfg = record.get("state", {}).get("day1_config")
        if cfg is not None:
            return cfg
    return None


def _most_common_day1_config(results):
    """
    (sy, ps, hn), count) for whichever Day-1 board configuration
    appeared most often across `results` -- ties broken by whichever
    sorts first, which is fine for a display-only convenience label.
    Returns (None, 0) if no game carried a day1_config at all (should
    not normally happen once day_phase(1) has run, but this is display
    code, not something worth crashing training over).
    """
    counts = defaultdict(int)
    for r in results:
        cfg = _extract_day1_config(r)
        if cfg is not None:
            counts[cfg] += 1
    if not counts:
        return None, 0
    best_cfg = max(counts, key=lambda cfg: (counts[cfg], cfg))
    return best_cfg, counts[best_cfg]


def _top_detailed_day1_configs(results, day1_config, top_n=3):
    """
    Among `results` whose (sy,ps,hn) COUNT matches `day1_config` (e.g.
    whatever _most_common_day1_config just picked out), break that
    single count bucket down further into its actual real/Werewolf/
    Madman compositions (board_split.format_day1_config_detailed) and
    return the `top_n` most common ones as (label, count) pairs,
    descending by count -- 2026-10-02 user request: the epoch-progress
    display used to show only the (sy,ps,hn) COUNT ("真狼-真狼-真狼",
    993/1000), which can't say whether a given "狼" slot across those
    993 games was usually an actual Werewolf or the Madman faking it
    (different games sharing the same count can -- and do -- have
    completely different true compositions). This answers that, for
    whichever handful of compositions are actually common within the
    bucket.

    Relies on each result carrying "day1_claims" and "roles" (added to
    main.run_game()'s return value alongside this function, 2026-10-02)
    -- a result missing either key (e.g. from some OTHER game-running
    path that hasn't been updated to include them) is silently skipped
    rather than crashing, since this is display-only convenience code.

    Returns a list of up to `top_n` (label, count) tuples -- fewer than
    `top_n` if the bucket doesn't actually contain that many distinct
    compositions, or [] if day1_config is None or no matching result
    carried the data needed to classify it.
    """
    if day1_config is None:
        return []

    counts = defaultdict(int)
    for r in results:
        if _extract_day1_config(r) != day1_config:
            continue
        claims = r.get("day1_claims")
        roles = r.get("roles")
        if claims is None or roles is None:
            continue
        label = board_split.format_day1_config_detailed(claims, roles)
        counts[label] += 1

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return ranked[:top_n]


def _format_top_detailed_day1_configs(results, day1_config, top_n=3):
    """
    `_top_detailed_day1_configs` rendered as the one-line, count-
    annotated string the epoch-progress display and run_until*.py use,
    e.g. "真狼-真狼-真狼 412件 / 真狂-真狼-真狼 301件 / 真狼-真狂-真狼
    180件". Falls back to the plain count-based notation (e.g.
    "(真狼-真狼-真狼)") if no detailed breakdown could be computed --
    either day1_config is None, or none of `results` carried the
    "day1_claims"/"roles" data _top_detailed_day1_configs needs.
    """
    top = _top_detailed_day1_configs(results, day1_config, top_n=top_n)
    if not top:
        return f"({board_split.format_day1_config(day1_config)})"
    return " / ".join(f"{label} {count}件" for label, count in top)


def find_matching_games(day1_config=None, detailed_label=None, winner=None,
                         n=3, max_games=50000, verbose_progress=True):
    """
    2026-10-02 user request: the epoch-progress display can show that
    a board shape like "真狼-真狼-真狂" is currently a near-total
    Werewolf win, but that number alone doesn't say HOW the wolves are
    winning it -- answering that means actually reading a handful of
    such games move-by-move. This scans FRESH games under the
    CURRENTLY LOADED policy weights (persistence.load_weights() first,
    as usual) until `n` of them match the given filter(s), and returns
    each match as a random.getstate() SNAPSHOT (not the game result
    itself) -- pass each one to replay_game() to print that EXACT
    game's full verbose log afterward. Snapshotting the random state
    instead of trying to re-print from the already-collected `history`
    is what makes this work: main.run_game()'s announce_co/print calls
    are scattered all through the game loop and were never written to
    render from a finished history list, but Python's `random` module
    (the ONLY source of randomness anywhere in this codebase -- see
    this function's own verification note below) is fully determined
    by its state, so replaying from the exact pre-game state and
    re-running with verbose=True reproduces an IDENTICAL game, move
    for move, as long as the policy weights haven't changed in
    between (don't call train()/persistence.save_weights() with new
    data between finding and replaying).

    Parameters
    ----------
    day1_config : (sy, ps, hn) tuple, optional
        Match games whose Day-1 claim COUNT equals this (board_split's
        usual bucket key).
    detailed_label : str, optional
        Match games whose exact real/Werewolf/Madman breakdown
        (board_split.format_day1_config_detailed) equals this string,
        e.g. "真狼-真狼-真狂" -- a STRICTER filter than day1_config
        alone, since many different true-role assignments share the
        same count. Pass both together to require both (day1_config
        alone is cheaper to check, so it's checked first).
    winner : "Village" or "Werewolf", optional
        Only match games this side won -- e.g. winner="Werewolf" to
        specifically find the wolf-favored games behind a board shape
        that's currently winning almost every time.
    n : int
        Stop scanning once this many matches are found.
    max_games : int
        Give up after scanning this many games (returning however
        many matches were found so far, possibly fewer than `n`) --
        protects against a combination so rare it would otherwise
        scan for a very long time.
    verbose_progress : bool
        Print a one-line "scanned so far / found so far" progress
        update periodically (every 2000 games) -- this scan can take
        a while for a rare combination, and silent output for minutes
        reads as a hang.

    Returns
    -------
    list of random.getstate() snapshots, one per matching game, in
    the order found. This function itself never prints a game's log
    or trains on the games it looks at -- every scanned game's result
    is discarded immediately after checking the filter.
    """
    matches = []
    scanned = 0
    for _ in range(max_games):
        state = random.getstate()
        result = main.run_game(verbose=False)
        scanned += 1

        if winner is not None and result["winner"] != winner:
            pass
        elif day1_config is not None and _extract_day1_config(result) != day1_config:
            pass
        elif detailed_label is not None and board_split.format_day1_config_detailed(
            result.get("day1_claims", []), result.get("roles", [])
        ) != detailed_label:
            pass
        else:
            matches.append(state)
            if len(matches) >= n:
                break

        if verbose_progress and scanned % 2000 == 0:
            print(f"[find_matching_games] scanned {scanned}/{max_games}, "
                  f"found {len(matches)}/{n}...")

    print(f"[find_matching_games] done: scanned {scanned} games, "
          f"found {len(matches)}/{n} matches.")
    return matches


def replay_game(state, show_role_candidates=False):
    """
    Re-run ONE game from a random.getstate() snapshot returned by
    find_matching_games(), with verbose=True, printing that EXACT
    game's full move-by-move log (CO order, votes, execution reasoning
    labels, night attack, etc.) so it can actually be read, not just
    counted.

    Requires the SAME policy weights that were loaded when find_
    matching_games() scanned for this snapshot -- training in between
    finding and replaying feeds the same random draws into DIFFERENT
    softmax probabilities, which diverges into a different game
    instead of reproducing the one that was found. If in doubt, just
    find and replay in the same uninterrupted script/session, right
    after persistence.load_weights().
    """
    random.setstate(state)
    return main.run_game(verbose=True, show_role_candidates=show_role_candidates)


def print_wolf_co_by_day(results):
    """
    今、朝一CO(wolf_ai.co_policy)が実際に「何日目」に「どの選択肢」を
    選んでいるかを、シミュレーション結果から集計して表示する。

    重みの一覧だけでは「revealed_Ps(霊能が既に名乗っている)」のような
    文脈特徴量がDay1では常に0(ブラインド)、Day2以降でだけ意味を持つ
    ことが分かりにくいので、実際の使用日を直接見えるようにするための
    もの。
    """
    day_option_counts = defaultdict(lambda: defaultdict(int))
    for r in results:
        for rec in r["history"]:
            if rec["side"] == "wolf" and rec["type"] == "co_action":
                day_option_counts[rec["day"]][rec["action"]] += 1

    print()
    print("=" * 78)
    print("狼AI: 朝一CO、実際は何日目に何を選んでいるか")
    print("=" * 78)

    if not day_option_counts:
        print("  (このバッチではCO判断が一度も発生しませんでした)")
        return

    for day in sorted(day_option_counts):
        counts = day_option_counts[day]
        total = sum(counts.values())
        parts = []
        for option in wolf_ai.CO_OPTIONS:
            c = counts.get(option, 0)
            pct = c / total * 100 if total else 0.0
            parts.append(f"{_wolf_co_option_jp(option)}={c}件({pct:.0f}%)")
        print(f"  {day}日目 (計{total}件): " + "  ".join(parts))

    print()
    print("  ※ Day1は「他人のCOが一切見えないブラインド状態」での選択、"
          "Day2以降は「既に公開されている情報を見た上」での選択(前日までに"
          "黙っていた狼が対象)。同じ重みを共有しているが、実際に使われる"
          "文脈(revealed_Ps等)はDay2以降でないと1にならない。")


def print_village_execution_accuracy(results):
    """
    村側(処刑投票)の「成果」--実際に処刑した相手が本当に狼だった割合
    --を、日ごと・全体で集計して表示する。重みが動いたかどうかだけで
    なく、その結果ちゃんと村が機能しているかを直接示すためのもの。
    """
    day_stats = defaultdict(lambda: [0, 0])  # day -> [狼を処刑できた数, 処刑総数]
    total_hits, total_execs = 0, 0

    for r in results:
        roles = r["roles"]
        for day_idx, target in enumerate(r.get("execution_history", []), start=1):
            is_wolf = roles[target] == "Wo"
            day_stats[day_idx][0] += 1 if is_wolf else 0
            day_stats[day_idx][1] += 1
            total_hits += 1 if is_wolf else 0
            total_execs += 1

    print()
    print("=" * 78)
    print("村AI: 処刑投票の成果(処刑した相手が本当に狼だったか)")
    print("=" * 78)

    if total_execs == 0:
        print("  (このバッチでは処刑が一度も発生しませんでした)")
        return

    for day in sorted(day_stats):
        hits, cnt = day_stats[day]
        print(f"  {day}日目の処刑 (計{cnt}件): 狼を処刑できた割合 {hits / cnt * 100:.1f}%")

    print(f"\n  全体: {total_hits}/{total_execs}件 "
          f"({total_hits / total_execs * 100:.1f}%) が実際に狼だった "
          f"(ランダムに選んだ場合の理論値は概ね2/7≒28.6%前後からゲーム進行に"
          f"つれて上昇していく)")


def print_hunter_co_by_day(results):
    """
    狩人AI: 朝一CO(名乗る/黙る)が実際に何日目に発生しているかの内訳。
    「処刑されそうになっての緊急CO」(progression_ai)は、通常の朝一CO
    とは別の判断ポイントなので分けて集計する。
    """
    day_counts = defaultdict(lambda: defaultdict(int))
    emergency_counts = defaultdict(int)

    for r in results:
        for rec in r["history"]:
            if rec["side"] == "hunter" and rec["type"] == "co_action":
                if rec["state"].get("under_threat"):
                    emergency_counts[rec["action"]] += 1
                else:
                    day_counts[rec["day"]][rec["action"]] += 1

    print()
    print("=" * 78)
    print("狩人AI: 朝一CO、実際は何日目に何を選んでいるか")
    print("=" * 78)

    if not day_counts and not emergency_counts:
        print("  (このバッチではCO判断が一度も発生しませんでした)")
        return

    for day in sorted(day_counts):
        counts = day_counts[day]
        total = sum(counts.values())
        reveal = counts.get("reveal", 0)
        silent = counts.get("silent", 0)
        print(f"  {day}日目 (計{total}件): "
              f"名乗る={reveal}件({reveal / total * 100:.0f}%)  "
              f"黙る={silent}件({silent / total * 100:.0f}%)")

    if emergency_counts:
        total_e = sum(emergency_counts.values())
        reveal_e = emergency_counts.get("reveal", 0)
        silent_e = emergency_counts.get("silent", 0)
        print(f"\n  緊急CO(処刑されそうになった時) (計{total_e}件): "
              f"名乗って助かろうとした={reveal_e}件({reveal_e / total_e * 100:.0f}%)  "
              f"黙って賭けに出た={silent_e}件({silent_e / total_e * 100:.0f}%)")


def print_hunter_guard_accuracy(results):
    """
    狩人AI: 護衛が実際に狼の襲撃を防げた割合(=護衛先と、その夜の狼の
    襲撃先が一致した割合)を、日ごと・全体で集計して表示する。
    """
    day_stats = defaultdict(lambda: [0, 0])  # day -> [的中数, 護衛実施数]
    total_hits, total_guards = 0, 0

    for r in results:
        guard_by_day = {}
        attack_by_day = {}
        for rec in r["history"]:
            if rec["side"] == "hunter" and rec["type"] == "guard":
                guard_by_day[rec["day"]] = rec["action"]
            elif rec["side"] == "wolf" and rec["type"] == "attack":
                attack_by_day[rec["day"]] = rec["action"]

        for day, guard_target in guard_by_day.items():
            if day not in attack_by_day:
                continue
            hit = guard_target == attack_by_day[day]
            day_stats[day][0] += 1 if hit else 0
            day_stats[day][1] += 1
            total_hits += 1 if hit else 0
            total_guards += 1

    print()
    print("=" * 78)
    print("狩人AI: 護衛の成果(実際に狼の襲撃を防げたか)")
    print("=" * 78)

    if total_guards == 0:
        print("  (このバッチでは護衛が一度も発生しませんでした)")
        return

    for day in sorted(day_stats):
        hits, cnt = day_stats[day]
        print(f"  {day}日目の護衛 (計{cnt}件): 襲撃を防げた割合 {hits / cnt * 100:.1f}%")

    print(f"\n  全体: {total_hits}/{total_guards}件 "
          f"({total_hits / total_guards * 100:.1f}%) で襲撃を防げた")


def print_medium_nomination_stats(results):
    """
    霊能AI: 灰指定(nomination_strategy)の「指定する率」と、実際に
    指定した相手が狼だった割合。この決定は本物の霊能だけでなく、
    霊能を騙る狼のフェイク指定にも同じ方策が使われている点に注意。
    """
    nominate_count, silent_count = 0, 0

    total_nominations, wolf_hits = 0, 0
    for r in results:
        roles = r["roles"]
        for rec in r["history"]:
            if rec["side"] == "medium" and rec["type"] == "nomination_strategy":
                if rec["action"] == "nominate":
                    nominate_count += 1
                else:
                    silent_count += 1

        final_nominations = {}
        for rec in r["history"]:
            noms = rec["state"].get("nominations")
            if noms:
                final_nominations.update(noms)
        for target in final_nominations.values():
            total_nominations += 1
            if roles[target] == "Wo":
                wolf_hits += 1

    print()
    print("=" * 78)
    print("霊能AI: 灰指定の成果")
    print("=" * 78)

    total_decisions = nominate_count + silent_count
    if total_decisions == 0:
        print("  (このバッチでは指定の判断が一度も発生しませんでした)")
        return

    print(f"  指定する : {nominate_count}件 ({nominate_count / total_decisions * 100:.1f}%)")
    print(f"  指定しない: {silent_count}件 ({silent_count / total_decisions * 100:.1f}%)")

    if total_nominations > 0:
        print(f"\n  実際に指定した相手が狼だった割合: "
              f"{wolf_hits}/{total_nominations}件 ({wolf_hits / total_nominations * 100:.1f}%) "
              f"※対象は村AIと同じ疑わしさの特徴量に基づく学習済みの選択"
              f"(ランダム選択ではなくなった)")
    print("\n  ※ 「誰を指定するか」の方策は本物の霊能だけでなく、霊能を騙る狼の偽指定にも共用されている")


def print_seer_stats(results):
    """
    占い師AI: 名乗るタイミング(通常/緊急)の内訳と、実際に黒判定を
    出した回数のうち何割が本当に狼だったか(=占いの的中率)を集計する。

    NOTE: 本物の占い師は現在、霊能と同様に初日CO必須のルールに変更
    されたため(main.py's seer_day_co)、"co_action"の意思決定ログは
    もう一切記録されない(常に強制的に名乗るため、学習対象の自由な
    選択ではなくなった)。よってこの関数の「名乗るタイミング」集計
    セクションは常に0件になるのが正常な状態。占いの的中率(下段)は
    投票ログから独立して集計されるため、これまで通り意味を持つ。
    """
    day_counts = defaultdict(lambda: defaultdict(int))
    emergency_counts = defaultdict(int)
    total_investigations, wolf_hits = 0, 0
    total_reveals = 0

    for r in results:
        roles = r["roles"]
        for rec in r["history"]:
            if rec["side"] == "seer" and rec["type"] == "co_action":
                if rec["state"].get("under_threat"):
                    emergency_counts[rec["action"]] += 1
                else:
                    day_counts[rec["day"]][rec["action"]] += 1
                if rec["action"] == "reveal":
                    total_reveals += 1
            elif rec["side"] == "seer" and rec["type"] == "investigate":
                target = rec["action"]
                total_investigations += 1
                if roles[target] == "Wo":
                    wolf_hits += 1

    print()
    print("=" * 78)
    print("占い師AI: CO、実際は何日目に何を選んでいるか")
    print("=" * 78)

    if not day_counts and not emergency_counts:
        print("  (このバッチではCO判断が一度も発生しませんでした)")
    else:
        for day in sorted(day_counts):
            counts = day_counts[day]
            total = sum(counts.values())
            reveal = counts.get("reveal", 0)
            silent = counts.get("silent", 0)
            print(f"  {day}日目 (計{total}件): "
                  f"名乗る={reveal}件({reveal / total * 100:.0f}%)  "
                  f"黙る={silent}件({silent / total * 100:.0f}%)")

        if emergency_counts:
            total_e = sum(emergency_counts.values())
            reveal_e = emergency_counts.get("reveal", 0)
            silent_e = emergency_counts.get("silent", 0)
            print(f"\n  緊急CO(処刑されそうになった時) (計{total_e}件): "
                  f"名乗って助かろうとした={reveal_e}件({reveal_e / total_e * 100:.0f}%)  "
                  f"黙って賭けに出た={silent_e}件({silent_e / total_e * 100:.0f}%)")

    print(f"\n  ゲーム{len(results)}戦中、占い師が名乗った回数: {total_reveals}件")

    if total_investigations > 0:
        print(f"  占いが実際に狼を引き当てた割合: "
              f"{wolf_hits}/{total_investigations}件 "
              f"({wolf_hits / total_investigations * 100:.1f}%) "
              f"※このゲームの狼比率は2/9≒22.2%なので、これを上回っていれば"
              f"「未CO(灰)を優先する」等の学習が機能している")


def evaluate(num_games=1000, verbose=True):
    """
    A LARGE, separate measurement (no weight updates) to get a
    trustworthy win-rate estimate. Training itself deliberately
    uses small batches (see earlier discussion of why), but that
    means a single epoch's win rate is too noisy to report as
    "the" win rate -- e.g. even with weights completely frozen,
    repeated 50-game measurements alone swing between roughly
    20% and 45% by pure chance. Averaging many more games here
    shrinks that noise down to a trustworthy estimate.

    verbose=False (train() now uses this) skips the 6 detailed
    breakdown sections below (wolf CO timing, village execution
    accuracy, hunter CO timing, hunter guard accuracy, medium
    nomination stats, seer stats) -- these were a big chunk of why a
    train() run's console output got long enough to scroll past a
    terminal's scrollback/truncate (2026-10 chat). The win rate
    itself is still computed and returned either way; only the extra
    printing is skipped. Pass verbose=True (or call evaluate()
    directly) when you actually want those breakdowns.
    """
    results = self_play.run_self_play(num_games, verbose=False)
    wins = sum(1 for r in results if r["winner"] == "Village")
    rate = wins / num_games
    # rough 95% confidence interval for a proportion
    margin = 1.96 * (rate * (1 - rate) / num_games) ** 0.5
    if verbose:
        print_wolf_co_by_day(results)
        print_village_execution_accuracy(results)
        print_hunter_co_by_day(results)
        print_hunter_guard_accuracy(results)
        print_medium_nomination_stats(results)
        print_seer_stats(results)
    return rate, margin


DAY1_CONFIG_REPORT_GAMES = 2000


def day1_config_report(num_games=DAY1_CONFIG_REPORT_GAMES):
    """
    Village win rate broken down by the Day-1 CO configuration (how
    many players claimed each forced/optional Day-1 role): "占-霊-狩"
    means (Sy claimants, Ps claimants, Hn claimants), all counted via
    claim_day[i] == 1 so a later (Day 2+) claim of the same role never
    gets miscounted into this bucket.

    This runs its OWN fresh batch of games (main.run_game directly,
    not self_play.run_self_play) because the per-game claim_day/
    co_history snapshot this needs isn't part of what run_self_play's
    result dicts carry -- reading main.claim_day/main.co_history
    immediately after each main.run_game() call is the only place
    that information is still available before the next game's
    reset_game() overwrites it.

    Called automatically at the end of train() (see below), using a
    smaller default game count than evaluate()'s 1000 -- there are up
    to a few dozen distinct (占,霊,狩) buckets, so getting a stable
    per-bucket win rate for the COMMON configurations needs more total
    games than a single trustworthy headline number does; rare
    configurations will still show a small, noisy sample size, which
    is printed alongside the rate so it's clear how much to trust it.
    Can also be called standalone (without retraining) for a bigger
    sample, e.g.:
        python3 -c "import train_and_report as t; t.day1_config_report(10000)"
    """
    stats = {}  # (sy, ps, hn) -> [village_wins, total_games]

    for _ in range(num_games):
        result = main.run_game(verbose=False)
        claim_day = main.claim_day
        co_history = main.co_history

        sy = sum(1 for i, co in enumerate(co_history) if co == "Sy" and claim_day[i] == 1)
        ps = sum(1 for i, co in enumerate(co_history) if co == "Ps" and claim_day[i] == 1)
        hn = sum(1 for i, co in enumerate(co_history) if co == "Hn" and claim_day[i] == 1)

        key = (sy, ps, hn)
        counts = stats.setdefault(key, [0, 0])
        counts[1] += 1
        if result["winner"] == "Village":
            counts[0] += 1

    print()
    print("=" * 78)
    print(f"初日の盤面別・村勝率 (占-霊-狩の初日CO人数、{num_games}戦、件数の多い順)")
    print("=" * 78)
    print(f"{'占-霊-狩':>10} {'村勝率':>8} {'件数':>7}")
    for sy, ps, hn in sorted(stats.keys(), key=lambda k: -stats[k][1]):
        wins, total = stats[(sy, ps, hn)]
        label = board_split.format_day1_config((sy, ps, hn))
        print(f"{label:>10} {wins / total * 100:7.1f}% {total:7d}")

    return stats


def board_split_win_rate_report(num_games=10000):
    """
    Village win rate broken down by ACTUAL board_split bucket (not
    every raw (Sy,Ps,Hn) triple the way day1_config_report does) --
    this is the report to reach for when checking board_split.py's
    results, because it groups exactly the way village_ai.execution_
    split / wolf_ai.attack_split actually route decisions: rare
    configs collapse into OTHER_BUCKET unless board_split.
    enable_full_split() has been called, in which case every distinct
    config gets its own row here too, matching its own bucket.

    Also prints the CUMULATIVE number of training samples each
    bucket's policy has ever received (village_ai.execution_split.
    sample_counts / wolf_ai.attack_split.sample_counts, persisted
    across runs -- see board_split.BoardSplitManager.record_sample),
    right next to this call's own fresh win-rate sample size, so it's
    clear at a glance both "how much has this bucket actually
    trained on" and "how many games does THIS report's win rate rest
    on" -- two different numbers that are easy to conflate.

    Meaningless (prints a note and returns) before village_ai.
    execution_split.split_enabled is True -- with the split off,
    there's only the one shared bucket and day1_config_report already
    covers it.

    Can be called standalone for a bigger sample, same as
    day1_config_report:
        python3 -c "import persistence, train_and_report as t; \\
            persistence.load_weights(); t.board_split_win_rate_report(50000)"
    """
    if not village_ai.execution_split.split_enabled:
        print("\n(board_split はまだ有効化されていません -- enable_board_split() を"
              "呼ぶまでは共有方策1本なので、day1_config_report() の方を見てください)")
        return None

    stats = {}  # bucket_key -> [village_wins, total_games]

    for _ in range(num_games):
        result = main.run_game(verbose=False)
        day1_cfg = None
        for record in reversed(result["history"]):
            day1_cfg = record["state"].get("day1_config")
            if day1_cfg is not None:
                break
        key = board_split.bucket_key(day1_cfg)
        counts = stats.setdefault(key, [0, 0])
        counts[1] += 1
        if result["winner"] == "Village":
            counts[0] += 1

    def _bucket_name(key):
        if key == board_split.OTHER_BUCKET:
            return "その他(レア盤面まとめ)"
        return board_split.format_day1_config(key)

    print()
    print("=" * 78)
    print(f"盤面バケツ別・村勝率 (board_split の実際の分け方で集計、今回{num_games}戦、"
          "件数の多い順)")
    print("=" * 78)
    print(f"{'盤面':>18} {'村勝率':>8} {'今回の件数':>10} {'累計学習サンプル数':>18}")
    for key in sorted(stats.keys(), key=lambda k: -stats[k][1]):
        wins, total = stats[key]
        trained_n = village_ai.execution_split.sample_counts.get(key, 0)
        print(f"{_bucket_name(key):>18} {wins / total * 100:7.1f}% {total:10d} {trained_n:18d}")

    return stats


def train():
    persistence.load_weights()
    print(f"Cumulative before this run: {persistence.total_epochs} epochs "
          f"({persistence.total_games} games)")

    before_snapshot = _snapshot_all_weights()

    print()
    print(f"Training for {NUM_EPOCHS} more epochs "
          f"({GAMES_PER_EPOCH} games/epoch, lr={LEARNING_RATE})...")
    print("(per-epoch numbers below are NOISY -- small batch, for training only)")
    print()
    print(f"{'epoch':>6} {'village win rate (noisy)':>26}   most common 盤面(占-霊-狩)")

    _start_epochs, _start_games = persistence.total_epochs, persistence.total_games
    for epoch in range(1, NUM_EPOCHS + 1):
        # 2026-10-06: periodic checkpoint (atomic save) so a crash or
        # Ctrl-C mid-run no longer loses every epoch since the start.
        if epoch > 1 and (epoch - 1) % CHECKPOINT_EVERY == 0:
            persistence.total_epochs = _start_epochs + (epoch - 1)
            persistence.total_games = _start_games + (epoch - 1) * GAMES_PER_EPOCH
            persistence.save_weights()
        results = self_play.run_self_play(GAMES_PER_EPOCH, verbose=False)
        village_wins = sum(1 for r in results if r["winner"] == "Village")

        self_play.update_policies(results, learning_rate=LEARNING_RATE,
                                   use_value_function=USE_VALUE_FUNCTION,
                                   value_learning_rate=VALUE_LEARNING_RATE)
        self_play.decay_all_policies(decay_rate=DECAY_RATE)

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            best_cfg, best_count = _most_common_day1_config(results)
            if best_cfg is None:
                cfg_label = "(不明)"
            else:
                cfg_label = _format_top_detailed_day1_configs(results, best_cfg)
            print(f"{epoch:6d} {village_wins / GAMES_PER_EPOCH * 100:25.1f}%   {cfg_label}")

    persistence.total_epochs = _start_epochs + NUM_EPOCHS
    persistence.total_games = _start_games + NUM_EPOCHS * GAMES_PER_EPOCH
    persistence.save_weights()
    print(f"Cumulative after this run: {persistence.total_epochs} epochs "
          f"({persistence.total_games} games)")

    print()
    print("Evaluating on 1000 games (this number IS trustworthy)...")
    rate, margin = evaluate(1000, verbose=False)
    print(f"Village win rate: {rate*100:.1f}% (+/- {margin*100:.1f} points, 95% confidence)")

    # 2026-10チャットでの要望: train()実行後のコンソール出力は「盤面
    # バケツ別・村勝率」の表1つだけに絞ってほしい、それ以外(個別AIの
    # 学習前後比較、盤面バケツ別の重み一覧、初日の盤面別・村勝率の素朴
    # 集計)は全部消してOK。ファイルへの書き出し自体は全部そのまま
    # 残してある(write_*_change_report / write_board_split_report は
    # announce=False でも中身は書く)ので、詳細を見たい時は同じフォルダ
    # の *_change_report.txt / board_split_report.txt を直接開けば
    # これまで通りの内容が見られる。
    #
    # day1_config_reportは、まだboard_splitを有効化していない(=共有
    # 方策のまま)場合のフォールバックとしてだけ残してある -- その
    # 状態ではboard_split_win_rate_reportが「まだ有効化されていません」
    # としか言わず何も表示しないので、盤面別の内訳を見る手段がこれしか
    # ない。board_split有効化後は呼ばない(内容が丸かぶりするため)。
    if not village_ai.execution_split.split_enabled:
        day1_config_report()

    after_snapshot = _snapshot_all_weights()
    write_wolf_change_report(before_snapshot, after_snapshot, announce=False)
    write_village_change_report(before_snapshot, after_snapshot, verbose=False, announce=False)
    write_hunter_change_report(before_snapshot, after_snapshot, announce=False)
    write_medium_change_report(before_snapshot, after_snapshot, announce=False)
    write_seer_change_report(before_snapshot, after_snapshot, announce=False)
    write_madman_change_report(before_snapshot, after_snapshot, announce=False)

    if village_ai.execution_split.split_enabled or wolf_ai.attack_split.split_enabled:
        write_board_split_report(announce=False)
        board_split_win_rate_report(DAY1_CONFIG_REPORT_GAMES)


# ============================================================
# Per-Day-1-board-config weights (see board_split.py). Manual,
# one-way switch-over from shared to per-board weights, plus a
# report so each bucket's learned differences from the shared
# baseline stay visible the same way every other policy's does.
# ============================================================

BOARD_SPLIT_REPORT_PATH = "board_split_report.txt"


def enable_board_split():
    """
    ONE-WAY switch-over from a single shared policy to per-Day-1-
    board-config weight sets, for village_ai.execution_policy
    (execution voting) and wolf_ai.attack_policy (night attack
    target) -- see board_split.py for the full mechanism and
    board_split.COMMON_DAY1_CONFIGS for which board configurations
    get their own bucket (everything else shares "other").

    This is a DELIBERATELY MANUAL step, not something train()
    triggers on its own: call it once, whenever the shared policies'
    win rate looks like it's stopped moving much run over run (the
    project's own call: total_epochs=66303 / total_games=3,315,060+
    as of 2026-09 was judged enough headroom -- see the chat
    discussion this shipped with). Persists immediately, so the
    switch survives a restart even without an explicit train() run
    afterward:

        python3 -c "import persistence, train_and_report as t; \\
            persistence.load_weights(); t.enable_board_split(); \\
            persistence.save_weights()"

    After this, every subsequent train() run keeps training (and
    reporting on) whichever per-bucket policies actual games touch --
    the shared policy itself is frozen from this point on (see
    board_split.BoardSplitManager's docstring).
    """
    # 2026-10 "split everything" extension: originally this only
    # covered execution_split/attack_split (the two policies board_
    # split.py's module docstring used to call out as the ONLY ones
    # split this way). Per the chat discussion that motivated this
    # extension -- a board config's execution/attack buckets could
    # still collapse even once fully protected from decay/cross-
    # config contamination, because every OTHER decision-making
    # policy (and the shared value-function baseline) kept co-evolving
    # out from under it -- every eligible policy across every AI
    # module is now switched over together, so "enable board split"
    # means the whole decision-making stack, not just these two.
    #
    # NOT included (2026-10 user decision, chat discussion): wolf_ai.
    # co_policy / hunter_ai.co_policy / madman_ai.co_policy (the
    # reveal-or-stay-silent timing decisions) and wolf_ai.day1_joint_
    # co_policy -- all four are part of what PRODUCES day1_config in
    # the first place, so splitting them BY day1_config would be
    # reasoning backwards from effect to cause. They stay single
    # shared policies; see board_split.py's module docstring.
    #
    # 2026-10-02 FIX: the first attempt at "split everything" put all
    # 19 eligible managers under the SAME global FULL_SPLIT_MODE flag
    # village_ai.execution_split/wolf_ai.attack_split already used --
    # which meant the 17 NEWLY split managers immediately fragmented
    # into ~30 buckets each, with only ~165 epochs/165,000 games behind
    # them. That's nowhere near enough for even one policy to converge
    # per bucket, let alone 17 simultaneously -- confirmed by an A/B
    # test against the pre-extension backup, which showed this crashed
    # the village win rate from ~73% to 0.5%. The fix (this user's
    # "共有でしばらく試した後その数値を個別バケツに入れ、その後共有数値
    # を個別で改変していく" instruction, 2026-10-02 chat): a manager's
    # full-split behavior is now an independent, per-manager choice
    # (BoardSplitManager.force_full_split -- see board_split.py). The
    # two PROVEN managers keep following the global FULL_SPLIT_MODE
    # flag (months of training behind their full split already); the
    # 17 newly (re-)split managers are explicitly pinned to the far
    # less data-hungry common-~10-configs-only grouping regardless of
    # the global flag, so splitting them doesn't repeat the same
    # dilution collapse. Each one still warm-starts its buckets from
    # whatever the shared policy has ALREADY learned over 66,000+
    # epochs of shared training (the "share for a while first" half of
    # the user's instruction, already satisfied by this point) and
    # trains each bucket independently from there onward (the "modify
    # the shared values individually per bucket" half).
    PROVEN_FULL_SPLIT_MANAGERS = [village_ai.execution_split, wolf_ai.attack_split]
    COMMON_ONLY_SPLIT_MANAGERS = [
        village_ai.guard_instruction_split,
        wolf_ai.vote_split,
        wolf_ai.fake_declaration_split, wolf_ai.fake_report_split,
        wolf_ai.fake_seer_report_split, wolf_ai.accuse_target_split,
        wolf_ai.endgame_declare_split,
        medium_ai.nomination_split, medium_ai.nomination_target_split,
        hunter_ai.guard_split,
        seer_ai.investigate_split,
        villager_ai.endgame_bluff_split,
        madman_ai.vote_split, madman_ai.fake_report_split,
        madman_ai.fake_seer_report_split, madman_ai.accuse_target_split,
        madman_ai.endgame_reveal_split,
    ]
    for manager in PROVEN_FULL_SPLIT_MANAGERS:
        manager.enable_split()
    for manager in COMMON_ONLY_SPLIT_MANAGERS:
        manager.enable_split()
        manager.force_full_split(False)
    total = len(PROVEN_FULL_SPLIT_MANAGERS) + len(COMMON_ONLY_SPLIT_MANAGERS)
    print(f"[board_split] switched {total} policies across every AI module "
          "from one shared policy each to per-Day-1-board-config weight "
          "sets (wolf_ai.day1_joint_co_policy and every side's CO reveal-"
          "timing policy stay shared -- they're what PRODUCES day1_config, "
          "so splitting BY day1_config can't apply to them). "
          f"{len(PROVEN_FULL_SPLIT_MANAGERS)} (execution/attack) follow the "
          f"global full-split setting; the other {len(COMMON_ONLY_SPLIT_MANAGERS)} "
          "are pinned to the common-~10-configs-only grouping regardless of "
          "it, to avoid repeating the 2026-10-01 dilution collapse. Each "
          "shared policy is now frozen; every board config learns from here "
          "via its own bucket, warm-started from the shared policy's "
          "current weights.")


def enable_full_board_split():
    """
    EXPERIMENTAL: widen village_ai.execution_split / wolf_ai.
    attack_split from the default "~10 common configs + one shared
    'other' bucket" to give EVERY distinct Day-1 board configuration
    its own bucket -- board_split.enable_full_split() does the actual
    switch (read its docstring for why this is explicitly a "let's
    measure it" tool, not the recommended default rollout). Also
    turns on ordinary split mode first via enable_board_split(), in
    case it wasn't on yet, so this can be called standalone.

    Recommended usage for testing: run this against a SEPARATE copy
    of trained_weights.pkl, not the main one -- both this and
    enable_board_split() are ONE-WAY switches, and the rarest configs
    (day1_config_report's long tail has entries below 0.05% of games)
    will need a very large number of additional games before their
    own bucket has anywhere near enough samples to judge, e.g.:

        cp trained_weights.pkl trained_weights_full_split_test.pkl
        python3 -c "
    import persistence, train_and_report as t
    persistence.load_weights('trained_weights_full_split_test.pkl')
    t.enable_full_board_split()
    persistence.save_weights('trained_weights_full_split_test.pkl')
    "

    then point a training run at that same path (persistence.
    load_weights/save_weights both take a `path` argument) to keep
    experimenting without touching the main trained_weights.pkl.
    """
    enable_board_split()
    board_split.enable_full_split()
    print("[board_split] widened to FULL per-board-config splitting for "
          "village_ai.execution_split / wolf_ai.attack_split -- every "
          "distinct Day-1 (Sy,Ps,Hn) combination now gets its own bucket "
          "for those two (previously, anything below ~1% frequency shared "
          "'other'). The global FULL_SPLIT_MODE flag this sets no longer "
          "widens every other split policy the way it used to (2026-10-02 "
          "fix) -- the other policies enable_board_split() just switched on "
          "are explicitly pinned to the common-~10-configs-only grouping "
          "via force_full_split(False), regardless of this flag, to avoid "
          "repeating the dilution collapse documented in board_split.py's "
          "module docstring. To widen one of THOSE to full splitting too, "
          "call manager.force_full_split(True) on it directly once it has "
          "enough data (see board_split.BoardSplitManager.force_full_split).")


def promote_all_splits_to_full():
    """
    One-way: flips every policy enable_board_split() pinned to the
    common-~10-configs-only mode (the 17 COMMON_ONLY_SPLIT_MANAGERS)
    over to FULL per-config splitting -- the same mode village_ai.
    execution_split / wolf_ai.attack_split already use. After this,
    there is no OTHER_BUCKET grouping left anywhere in the 19 eligible
    policies: every distinct (Sy,Ps,Hn) Day-1 configuration that is
    actually played gets, and keeps, its own independent weight set
    for every one of them.

    2026-10-02 user decision (chat discussion): the correct line of
    play differs enough by board shape that lumping rare configs
    together under OTHER_BUCKET was judged worse than letting each one
    train on its own, however little data it gets -- accepting the
    "reverting to common-only is a one-way, data-hungry commitment"
    cost described in BoardSplitManager.force_full_split's docstring,
    with no plan (for now) to revert. warm-starting still applies: a
    config that peels off OTHER_BUCKET the next time it's played
    starts from OTHER_BUCKET's current weights (see
    BoardSplitManager.active), not from scratch.

    Call this ONCE against the live weights file, e.g.:

        python3 -c "import persistence, train_and_report as t; \\
            persistence.load_weights(); t.promote_all_splits_to_full(); \\
            persistence.save_weights()"

    then keep training as usual -- nothing else about train()/
    run_self_play() changes.
    """
    managers = [
        village_ai.execution_split, wolf_ai.attack_split,
        village_ai.guard_instruction_split,
        wolf_ai.vote_split, wolf_ai.fake_declaration_split, wolf_ai.fake_report_split,
        wolf_ai.fake_seer_report_split, wolf_ai.accuse_target_split, wolf_ai.endgame_declare_split,
        medium_ai.nomination_split, medium_ai.nomination_target_split,
        hunter_ai.guard_split, seer_ai.investigate_split, villager_ai.endgame_bluff_split,
        madman_ai.vote_split, madman_ai.fake_report_split, madman_ai.fake_seer_report_split,
        madman_ai.accuse_target_split, madman_ai.endgame_reveal_split,
    ]
    for m in managers:
        if not m.split_enabled:
            m.enable_split()
        m.force_full_split(True)
    print(f"[board_split] promoted {len(managers)} policies to full per-config "
          "splitting -- OTHER_BUCKET grouping is no longer used by any of "
          "them; every distinct Day-1 board configuration that appears gets "
          "its own bucket, warm-started from whatever OTHER_BUCKET (or the "
          "frozen shared base) had already learned for it.")


def _board_split_bucket_name(key):
    if key == board_split.OTHER_BUCKET:
        return "その他(レア盤面まとめ)"
    return board_split.format_day1_config(key)


def print_top_board_split_buckets(manager, labels, title, top_n=2, diff_threshold=0.05, max_lines=15):
    """
    train()がコンソールに表示する「隠れ層」の代わり。board_split有効化後は
    manager.base(=村AIならvillage_ai.execution_policy)は凍結されて二度と
    更新されないので、これを表示し続けても実際の判断内容を反映しなくなる
    (2026-10チャットで指摘された問題)。代わりに、実際に使われている
    盤面バケツのうち学習サンプル数が多いもの上位`top_n`個だけ、共有
    ベースラインからの差分が大きい特徴量だけをコンソールに表示する --
    全44項目x全バケツを出すと今度はそれはそれで長すぎるので、変化が
    目立つものだけに絞ってある。全件・全バケツを見たい時は
    board_split_report.txt(write_board_split_reportが書き出す)を
    直接開く。
    """
    if not manager.split_enabled:
        return
    print()
    print("=" * 78)
    print(f"{title} -- 盤面バケツ別の重み(学習サンプル数が多い順、上位{top_n}バケツ、"
          f"共有ベースからの差分が目立つ特徴量のみ)")
    print("=" * 78)
    if not manager.buckets:
        print("(まだどの盤面バケツも実際のゲームで使われていません)")
        return

    base_weights = manager.base.weights
    ordered = sorted(
        manager.buckets.items(),
        key=lambda kv: manager.sample_counts.get(kv[0], 0),
        reverse=True,
    )
    for key, policy in ordered[:top_n]:
        n = manager.sample_counts.get(key, 0)
        print(f"\n[盤面: {_board_split_bucket_name(key)}]  (累計学習サンプル数: {n}件)")
        diffs = []
        for label, w, base_w in zip(labels, policy.weights, base_weights):
            diff = w - base_w
            if abs(diff) >= diff_threshold:
                diffs.append((abs(diff), label, w, diff))
        diffs.sort(reverse=True)
        if not diffs:
            print(f"  (共有ベースから差分{diff_threshold}以上の特徴量なし)")
        for _, label, w, diff in diffs[:max_lines]:
            print(f"  {label:<45}: {w:+.3f}  (共有比 {diff:+.3f})")
        if len(diffs) > max_lines:
            print(f"  ...他{len(diffs) - max_lines}件(全件は board_split_report.txt 参照)")
    print(f"\n(全バケツ・全特徴量は {BOARD_SPLIT_REPORT_PATH} に保存されています)")


def _board_split_bucket_report(manager, labels, title):
    lines = [f"=== {title} ===", ""]
    lines.append(f"分岐状態: {'有効' if manager.split_enabled else '無効(共有方策のまま)'}")
    lines.append("")

    base_weights = manager.base.weights
    lines.append("[共有方策(分岐前のベース、以後は凍結)]")
    for label, w in zip(labels, base_weights):
        lines.append(f"  {w:+.3f}  {label}")
    lines.append("")

    if not manager.buckets:
        lines.append("(まだどの盤面バケツも実際のゲームで使われていません)")
        return "\n".join(lines)

    def _bucket_name(key):
        if key == board_split.OTHER_BUCKET:
            return "その他(レア盤面まとめ)"
        return board_split.format_day1_config(key)

    # Most-trained bucket first -- this is exactly the answer to "how
    # much can I trust this bucket's weights yet", so it belongs at
    # the top, not buried alphabetically among rare ones.
    ordered = sorted(
        manager.buckets.items(),
        key=lambda kv: manager.sample_counts.get(kv[0], 0),
        reverse=True,
    )
    for key, policy in ordered:
        n = manager.sample_counts.get(key, 0)
        lines.append(f"[盤面: {_bucket_name(key)}]  (これまでの学習サンプル数: {n}件)")
        for label, w, base_w in zip(labels, policy.weights, base_weights):
            diff = w - base_w
            marker = " ← 共有方策から変化" if abs(diff) >= 0.05 else ""
            lines.append(f"  {w:+.3f}  (共有比 {diff:+.3f})  {label}{marker}")
        lines.append("")

    return "\n".join(lines)


def write_board_split_report(path=BOARD_SPLIT_REPORT_PATH, announce=True):
    """
    Per-bucket weights for both board-split policies, each line also
    showing how far that bucket has drifted from the shared baseline
    it was warm-started from -- the board-split equivalent of the
    other write_*_change_report functions above (a diff makes it
    obvious which board-specific behaviors actually emerged, instead
    of having to eyeball two full weight dumps side by side).

    announce=False (train()が使う、2026-10): "wrote path"の1行すら
    出さない -- train()のコンソール出力を「盤面バケツ別・村勝率」の
    表だけに絞ってほしいというチャットでの要望への対応。ファイルへの
    書き出し自体は announce の値に関わらず常に行われる。
    """
    sections = [
        _board_split_bucket_report(
            village_ai.execution_split, _village_execution_labels(), "処刑投票(盤面別)"
        ),
        _board_split_bucket_report(
            wolf_ai.attack_split, WOLF_ATTACK_LABELS, "人狼の襲撃先(盤面別)"
        ),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections))
    if announce:
        print(f"[board_split] wrote {path}")


# ============================================================
# Weight reports (replaces the old table-based situation listing --
# now that village_ai/hunter_ai are back to plain linear models,
# the weights themselves ARE the readable summary; no need to
# enumerate every situation by hand.)
# ============================================================

# 役職コード -> 日本語(ラベル組み立て用。config.ROLE_JP は盤面表示用の
# 1文字略称なので、レポート用にはこちらの読みやすい表記を別に持つ)。
_ROLE_NAME_JP = {"Ps": "霊能", "Hn": "狩人", "Sy": "占い師"}


def _village_execution_labels():
    labels = ["基本(バイアス)"]
    for role in village_ai.CLAIMABLE_ROLES:
        role_jp = _role_jp(role)
        labels += [
            f"対象が{role_jp}を名乗っている",
            f"対象が{role_jp}を名乗っている×名乗り人数",
            f"対象が唯一の{role_jp}CO",
            f"対象の{role_jp}COが確定で偽と証明されている",
            f"対象が{role_jp}を名乗っている場合の理論上の狼率((人数-1)/人数)",
            f"対象が{role_jp}CO(競合)で、対抗が全員死んで唯一の生存者",
        ]
    labels += [
        "対象が村人を名乗っている",
        "対象が確定で村人(白)",
        "対象が確定で偽の狩人",
        "対象が確定で本物の狩人",
        "生存人数",
        "これまでの護衛成功回数",
        "偽の狩人から「白」と庇われたことがある対象",
        "護衛指示の検証待ち状態",
        "唯一の霊能から指定されている対象",
        "対抗している霊能から指定されている対象",
        "ルール上あり得ない遅れた霊能CO",
        "確定した霊能結果と矛盾する対象",
        "残り狼数から見た圧力(最後の1匹に迫っている状況)",
        "処刑済みの人から「白」と庇われていた対象",
        "黒判定者に投票しなかった回数(不審投票カウント)",
        "後に潔白と判明した対抗霊能から指定されている対象",
        "投票者自身が指定した対象",
        "唯一の霊能の指定に従わなかった回数(制裁対象カウント)",
        "保護されるべき相手に投票した回数(狼らしい積極的な不審投票カウント)",
        "唯一の占い師から黒判定を受けている対象",
        "対抗している占い師から黒判定を受けている対象",
        "対象自身の占いCOが確定で矛盾(本物確定の占い師の結果と食い違う)",
        "本物と確定した占い師から黒判定を受けている対象",
        "対象が正体を明かしている(狂人の捨てCOまたは村騙り、区別はつかない)",
        "投票者自身(本物の占い師)が占いで黒と判定した対象",
        "対象が「狂人CO」を名乗っている",
        "対象が唯一の「狂人CO」",
        "対象が「狂人CO」を名乗っている場合の理論上の狼率((人数-1)/人数)",
        "対象が戦績のある占い師から黒判定を受けている",
        "対象が本物確定の占い師から黒判定(確定狼)",
        "ロラ線から外れた投票の回数",
        "同役職の別CO者が狼と判明している",
        "最後の1匹に迫る状況(CO由来)",
        # ---- extra tail (village_ai._extra_features) ----
        "確定白に投票した回数",
        "確定狼に投票した回数",
        "唯一の占い師から白を貰っている",
        "競合する占い師から白を貰っている",
        "本物確定の占い師から白を貰っている",
        "誰からも報告されていない灰",
        "占い・霊能の矛盾回数",
        "偽判定(後に否定された結果)の回数",
        "投票者の同役職のライバルである",
        "投票者自身が白と占った対象",
        "投票者自身が黒と占った対象",
        "終盤(生存<=5)×CO者",
        "終盤×唯一CO",
        "終盤×灰",
        "最終盤(生存<=3)×CO者",
        "最終盤×灰",
        "占い師全員が白で一致",
        "占い師全員が黒で一致",
        "占い師の結果が割れている",
        "確定偽の占い師からの黒",
        "確定偽の占い師からの白",
        "対抗霊能からの黒",
        "対抗霊能からの白",
        "逼迫×唯一CO",
        "逼迫×確定白",
        "逼迫×灰",
        "逼迫×CO者",
        "余裕×CO者",
        "狩人CO無し×灰",
        "灰率×未報告灰",
        "投票者の私的な占い/霊能結果との矛盾",
        "指名者自身の投票矛盾",
        "自分の黒に投票しなかった占い師",
        "霊能が白と主張した人への投票回数",
    ]
    assert len(labels) == village_ai.NUM_FEATURES, (len(labels), village_ai.NUM_FEATURES)
    return labels


_GUARD_TARGET_TYPE_JP = {
    "gray": "ランダムな灰", "medium": "霊能", "hunter": "もう一方の狩人CO", "seer": "占い師",
}


def _village_guard_instruction_labels():
    labels = []
    for option in village_ai.GUARD_INSTRUCTION_OPTIONS:
        a, b = option.split("_")
        labels.append(f"1人目:{_GUARD_TARGET_TYPE_JP[a]} / 2人目:{_GUARD_TARGET_TYPE_JP[b]}")
    return labels


def _role_jp(role_code):
    return _ROLE_NAME_JP.get(role_code, role_code)


def _wolf_co_option_jp(option):
    """CO_OPTIONS の1つ(例: "claim_Ps", "stay_silent")を日本語化。"""
    if option == "stay_silent":
        return "黙る"
    role_code = option[len("claim_"):]
    return f"{_role_jp(role_code)}を騙る"


def _wolf_co_context_jp(ctx):
    """文脈フラグの1つ(例: "revealed_Ps", "emergency_reveal")を日本語化。"""
    if ctx == "emergency_reveal":
        return "直前に緊急COがあった"
    if ctx == "under_threat":
        return "自分が処刑されそうな状況(土壇場の回避CO)"
    if ctx == "personally_nominated":
        return "自分が指定されている"
    if ctx.startswith("teammate_claimed_"):
        role_code = ctx[len("teammate_claimed_"):]
        return f"仲間(狼/狂人)が既に{_role_jp(role_code)}を名乗っている"
    role_code = ctx[len("revealed_"):]
    return f"{_role_jp(role_code)}が既に名乗っている(相手が仲間か本物かは区別しない)"


def _wolf_co_labels():
    context_names = (
        [f"revealed_{role}" for role in wolf_ai.CLAIMABLE_ROLES]
        + ["emergency_reveal", "under_threat", "personally_nominated"]
        + [f"teammate_claimed_{role}" for role in wolf_ai.CLAIMABLE_ROLES]
    )
    labels = [_wolf_co_option_jp(option) for option in wolf_ai.CO_OPTIONS]
    for option in wolf_ai.CO_OPTIONS:
        for ctx in context_names:
            labels.append(f"{_wolf_co_option_jp(option)}×{_wolf_co_context_jp(ctx)}")
    return labels


WOLF_ATTACK_LABELS = [
    "基本(バイアス)",
    "対象が霊能を騙っている",
    "対象が本物と判明している霊能(狼から見て消去法で分かる)",
    "対象が狩人を騙っている(狩人は常に本物)",
    "経過日数",
    "襲撃すると味方の対抗COが偽確定してしまう(墓穴を掘るリスク)",
    "対象が本物と判明している占い師(狼から見て消去法で分かる)",
    "墓穴リスク×対象が本物の霊能",
    "墓穴リスク×対象が狩人を騙っている",
    "墓穴リスク×対象が本物の占い師",
    "対象が本物の占い師から白判定を受けている(狼から見て消去法で分かる)",
    "護衛リスクあり×対象が狩人を騙っている(狩人を先に噛めば護衛を封じられる)",
    "護衛リスクあり×対象が本物の占い師(狩人が生きている間に占いを噛むと護衛で外れる恐れ)",
]
WOLF_VOTE_LABELS = [
    "基本(バイアス)",
    "対象が仲間の狼",
    "対象が霊能を騙っている",
    "対象が狩人を騙っている",
    "対象が確定で村人(白)と分かっている",
    "経過日数",
    "対象が本日の唯一の信頼できる指定先(指定に従うふりをする動機)",
    "対象が占い師を騙っている",
    "対象が正体を明かしている(狂人の捨てCOまたは村騙り、区別はつかない)",
]
WOLF_FAKE_DECLARATION_LABELS = ["仲間の狼を庇う指定にする", "ランダムな相手を指定する"]
WOLF_FAKE_REPORT_LABELS = ["本物の霊能の報告に同調する", "本物の霊能の報告に矛盾させる"]
WOLF_FAKE_SEER_REPORT_LABELS = [
    "対抗占い師を黒判定にする(騙り合いを仕掛ける)",
    "仲間の狼を白判定で庇う",
    "無関係な相手をランダムに黒判定にする",
    "無関係な相手をランダムに白判定にする",
]

# 「対抗占い師を黒判定にする」を選んだ後、対抗の中の誰を名指しするか
# (身内の狼は候補から除外済み -- wolf_ai.choose_accuse_target のコメント
# 参照)。狼AI・狂人AI共通のレイアウト。
ACCUSE_TARGET_LABELS = [
    "基本(バイアス)",
    "対象が既に確定で村人(白)と判定されている",
    "対象の初日CO日 ÷ 7 (古い確定COほど値が大きい)",
    "対抗が1人しかいない(選択の余地なし)",
    "生存人数 ÷ 9",
]

# 霊能AI: co_policy(いつ名乗るか)は main.py 側でDay1強制COに置き換わって
# おり、実際には一度も呼ばれていない未使用コード(medium_day_co のコメント
# 参照)。この重みはmedium_ai.py自体から削除済み(2026年のクリーン
# アップで対応)。実際に使われている nomination_policy(灰指定するか)
# だけを対象にする。
MEDIUM_NOMINATION_LABELS = ["灰を指定する", "指定しない"]

HUNTER_CO_LABELS = [
    "名乗る", "黙る",
    "名乗る×これまでの護衛成功回数", "黙る×これまでの護衛成功回数",
    "名乗る×灰(未CO)人数", "黙る×灰(未CO)人数",
    "名乗る×処刑されそうな状況", "黙る×処刑されそうな状況",
    "名乗る×自分が指定されている", "黙る×自分が指定されている",
]
HUNTER_GUARD_LABELS = [
    "基本(バイアス)", "対象が霊能を騙っている", "経過日数",
    "対象が占い師を騙っている", "対象が占い師から白判定を受けている",
]

# seer_ai.co_policy (CO timing) was removed from seer_ai.py in the
# same 2026 cleanup pass as medium_ai.co_policy above -- the real
# Seer has been forced to reveal on Day 1 for a while now, so this
# had zero call sites. Only the still-live investigate_policy remains.
SEER_INVESTIGATE_LABELS = [
    "基本(バイアス)", "既に占ったことがある相手", "対象が未CO(灰)", "経過日数",
]

# villager_ai.co_policy (the full-game "claim Medium/Hunter" village
# bluff, distinct from the endgame bluff below) was removed from
# villager_ai.py in the same cleanup pass -- main.py's day_phase had
# already stopped calling it (see main.py's day_phase comment), so it
# had zero call sites either.

def _endgame_context_labels():
    """
    Context values shared by madman_ai.choose_endgame_reveal_action
    and villager_ai.choose_endgame_bluff_action -- see village_ai.
    endgame_context_features's docstring for what each one means.
    """
    labels = [f"生存狼数={k}の確率" for k in range(village_ai.TOTAL_WOLVES + 1)]
    labels += ["残り3人の局面", "残り4人の局面", "既に役職(占/霊/狩)を騙っている"]
    return labels


def _endgame_decision_labels(option_labels):
    """
    Full interaction-feature label list for an N-option, context-
    interacted decision (wolf_ai's ENDGAME_DECLARE_OPTIONS / madman_
    ai's ENDGAME_REVEAL_OPTIONS / villager_ai's ENDGAME_BLUFF_OPTIONS,
    all 4 options since the "3-4-players-remain identity claim"
    redesign) -- one-hot option identity followed by context*option
    interaction terms, matching _endgame_declare_feature_list's /
    _endgame_reveal_feature_list's / _endgame_bluff_feature_list's
    exact layout. Currently unreachable in practice for these 3 keys
    (they're in _MATRIX_SPECS, so the matrix renderer is used instead
    -- see _endgame_matrix_spec), but kept correct for whichever N the
    caller's options list actually has, rather than hard-coded to 2.
    """
    labels = list(option_labels)
    for ctx_label in _endgame_context_labels():
        labels += [f"{ctx_label}×{opt}" for opt in option_labels]
    return labels


VALUE_FUNCTION_LABELS = [
    "bias", "alive_count", "is_medium_alive_and_revealed", "is_hunter_alive_and_revealed",
    "medium_claimant_count", "hunter_claimant_count", "guard_success_count",
    "confirmed_villagers_count", "confirmed_identified_wolves", "total_wolves",
    "day", "has_pending_verification", "is_seer_alive_and_revealed", "seer_claimant_count",
]


# ============================================================
# マトリクス形式レポート
#
# 「N択の選択肢(option)」×「文脈フラグ(context)」の一次結合+交互作用
# という同じ形をした方策がいくつもある(狼CO、狩人CO、占い師CO、村人の
# 騙りCO、最終盤の捨てCO/村騙り/正体表明)。これらは本質的に「要素が
# option/contextの2つだけ」の2次元データなので、フラットな1行1項目の
# リストで見せるより、option を列・context を行に置いたマトリクスの方が
# 全体像を一目で把握しやすい -- 同じcontextの中でoptionごとにどう傾向が
# 違うか(縦に比較)も、同じoptionの中でcontextごとにどう変わるか(横に
# 比較)も、フラットなリストだと拾いにくいが、マトリクスなら両方が
# 同時に見える。
#
# 各方策の実際の特徴ベクトルの並び順は2パターンある(方策ごとに固定):
#   - "option_major": option でブロック分けし、ブロック内で context が
#     動く。例: wolf_ai._co_feature_list, villager_ai._co_feature_list
#     (f[N_OPT + opt_i*N_CTX + ctx_i])
#   - "context_major": context でブロック分けし、ブロック内で option が
#     動く。例: hunter_ai._co_feature_list, seer_ai._co_feature_list,
#     village_ai.endgame_context_features を使う3つの最終盤方策
#     (wolf_ai._endgame_declare_feature_list, madman_ai._endgame_
#     reveal_feature_list, villager_ai._endgame_bluff_feature_list)
#     (f[N_OPT + ctx_i*N_OPT + opt_i])
# バイアス項(f[0..N_OPT-1])はどちらのパターンでも常に先頭 N_OPT 個で
# 共通。マトリクスでは「(基本バイアス)」という専用の行として先頭に
# 表示する。
# ============================================================

def _matrix_value_index(layout, n_opt, n_ctx, opt_i, ctx_i):
    """
    ctx_i == -1 はバイアス行(f[opt_i]そのもの)を指す特別扱い。
    """
    if ctx_i == -1:
        return opt_i
    if layout == "option_major":
        return n_opt + opt_i * n_ctx + ctx_i
    return n_opt + ctx_i * n_opt + opt_i  # "context_major"


def _format_weight_cell(value):
    return f"{value:+.3f}"


def _format_delta_cell(value):
    if abs(value) >= 0.5:
        marker = "**"
    elif abs(value) >= 0.05:
        marker = "*"
    else:
        marker = ""
    return f"{value:+.3f}{marker}"


def _display_width(text):
    """
    Monospace terminals render full-width characters (most CJK,
    including every Japanese label used here) as roughly TWICE as
    wide as an ASCII one. Plain str.ljust/rjust count each character
    as ONE column regardless, so padding a Japanese label the same
    way a numeric column is padded silently misaligns the whole
    table -- every row's numbers end up shifted relative to the
    header depending on how many wide characters happen to be in
    that row's label. This computes the terminal's actual column
    count instead of the character count, so the padding helpers
    below can line the matrix up correctly.
    """
    import unicodedata
    width = 0
    for ch in text:
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def _pad_left(text, width):
    """Left-justify `text` to `width` TERMINAL COLUMNS (not characters)."""
    return text + " " * max(0, width - _display_width(text))


def _pad_right(text, width):
    """Right-justify `text` to `width` TERMINAL COLUMNS (not characters)."""
    return " " * max(0, width - _display_width(text)) + text


def _render_matrix(option_headers, row_labels, value_fn, format_cell,
                    row_label_width=None, col_width=14):
    """
    option_headers : 列見出し(選択肢名)のリスト
    row_labels     : 行見出し(文脈名)のリスト -- "(基本バイアス)"行は
        別途、常に先頭に自動で追加されるので、ここには含めない。
    value_fn(ctx_i, opt_i) -> float
        ctx_i == -1 がバイアス行を意味する。

    列幅は _display_width で日本語の全角文字を2カラム分として数えて
    揃えているので、見出し行とデータ行がターミナル上で実際に縦に
    揃う(単純な str.ljust/rjust だと全角文字1文字を1カラムとしか
    数えないため、行によって見出しの見た目の長さが変わり、数値列が
    行ごとにズレて見えてしまう)。

    row_label_width : 省略時は、実際の行見出し(バイアス行込み)の中で
        最長のものに合わせて自動計算する -- 一部の行見出しに補足タグ
        (例: "[緊急CO時のみ]")が付いて長くなるケースでも、固定幅の
        決め打ちで途中から数値列がズレたり詰まったりしないようにする
        ため。
    """
    if row_label_width is None:
        row_label_width = max(
            [_display_width("(基本バイアス)")] + [_display_width(l) for l in row_labels]
        ) + 2

    lines = []
    header = _pad_left("", row_label_width) + "".join(
        _pad_right(h, col_width) for h in option_headers
    )
    lines.append(header)

    bias_cells = "".join(
        _pad_right(format_cell(value_fn(-1, oi)), col_width)
        for oi in range(len(option_headers))
    )
    lines.append(_pad_left("(基本バイアス)", row_label_width) + bias_cells)

    for ri, label in enumerate(row_labels):
        cells = "".join(
            _pad_right(format_cell(value_fn(ri, oi)), col_width)
            for oi in range(len(option_headers))
        )
        lines.append(_pad_left(label, row_label_width) + cells)

    return lines


def _render_symmetric_pair_matrix(base_options_jp, weight_lookup, format_cell,
                                   row_label_width=None, col_width=14):
    """
    Renders wolf_ai.day1_joint_co_policy's weights as the literal
    "wolf A's choice x wolf B's choice" 4x4 grid the user originally
    pictured, even though the underlying policy only has 10 weights
    (one per UNORDERED pair -- see wolf_ai.py's module comment on
    DAY1_JOINT_CO_OPTIONS for why unordered). The 4x4 grid is
    necessarily symmetric (cell [i][j] and [j][i] show the exact same
    weight, since "A claims Seer, B claims Medium" and "A claims
    Medium, B claims Seer" are the same underlying combination, just
    with the physical wolves swapped) -- that symmetry is expected,
    not a bug, and is exactly what makes 10 weights enough to fill a
    16-cell grid.

    base_options_jp : the 4 per-wolf option labels, in a fixed order
        used for both the rows and the columns.
    weight_lookup(a, b) -> float : returns the weight for the
        combination of option `a` and option `b` (order-independent).
    """
    if row_label_width is None:
        row_label_width = max(_display_width(l) for l in base_options_jp) + 2

    lines = []
    header = _pad_left("", row_label_width) + "".join(
        _pad_right(h, col_width) for h in base_options_jp
    )
    lines.append(header)

    for a in base_options_jp:
        cells = "".join(
            _pad_right(format_cell(weight_lookup(a, b)), col_width)
            for b in base_options_jp
        )
        lines.append(_pad_left(a, row_label_width) + cells)

    return lines


def _wolf_day1_joint_co_weight_lookup(weights):
    """
    Builds a symmetric (a, b) -> weight lookup from
    wolf_ai.day1_joint_co_policy's 10 unordered-pair weights, keyed
    by the JAPANESE option labels (so it can be called directly with
    _render_symmetric_pair_matrix's base_options_jp entries).
    """
    combo_weight = {}
    for (opt_a, opt_b), w in zip(wolf_ai.DAY1_JOINT_CO_OPTIONS, weights):
        # frozenset({x, x}) == frozenset({x}), so the "both pick the
        # same option" case collapses naturally -- no special-casing
        # needed for opt_a == opt_b.
        key = frozenset({_wolf_co_option_jp(opt_a), _wolf_co_option_jp(opt_b)})
        combo_weight[key] = w

    def lookup(a, b):
        return combo_weight[frozenset({a, b})]

    return lookup


def _wolf_co_context_liveness_tag(ctx):
    """
    Every context row in this matrix shares ONE weight vector across
    THREE different calling situations (main.py's wolf_day_co /
    madman_day_co routine check on Day 1, that same routine check on
    Day 2+, and the separate emergency evasion check that can fire on
    ANY day right after a nomination) -- but a given context flag is
    only EVER non-zero in a subset of those three, so its weight only
    ever receives gradient from that subset (see policy_utils.
    LinearPolicy.update: gradient_k = chosen_features[k] -
    expected_features[k], and both terms are exactly 0 whenever the
    context flag itself is 0 for every candidate). Tagging each row
    with which situation(s) can actually move it keeps the matrix
    from implying a wolf "consciously checks" something it
    structurally cannot see yet (e.g. revealed_Ps on Day 1, which
    build_public_state's blind=True forces to 0 no matter what
    actually happened) -- that row's weight is real and meaningful,
    it's just entirely a Day-2+/emergency-only signal, never a Day-1
    one.
    """
    if ctx in ("emergency_reveal", "under_threat", "personally_nominated"):
        # main.py's routine wolf_day_co/madman_day_co never sets these
        # keys at all (they default to False/absent) -- ONLY the
        # separate emergency-evasion check populates them.
        return " [緊急CO時のみ]"
    # revealed_* / teammate_claimed_*: forced to 0 on Day 1's routine
    # check (blind=True), but genuinely live on Day 2+'s routine
    # check AND on the emergency check (which is never blinded, even
    # on Day 1 -- see this conversation's earlier finding).
    return " [Day2以降の通常CO or 緊急CO時のみ ※Day1の通常COでは常に0]"


def _wolf_co_matrix_spec():
    option_headers = [_wolf_co_option_jp(option) for option in wolf_ai.CO_OPTIONS]
    context_names = (
        [f"revealed_{role}" for role in wolf_ai.CLAIMABLE_ROLES]
        + ["emergency_reveal", "under_threat", "personally_nominated"]
        + [f"teammate_claimed_{role}" for role in wolf_ai.CLAIMABLE_ROLES]
    )
    row_labels = [
        _wolf_co_context_jp(ctx) + _wolf_co_context_liveness_tag(ctx)
        for ctx in context_names
    ]
    return option_headers, row_labels, "option_major"


def _hunter_co_matrix_spec():
    option_headers = ["名乗る", "黙る"]
    row_labels = [
        "これまでの護衛成功回数", "灰(未CO)人数",
        "処刑されそうな状況", "自分が指定されている",
    ]
    return option_headers, row_labels, "context_major"


# Shared 4-option label list, in the exact order wolf_ai.ENDGAME_DECLARE_
# OPTIONS / madman_ai.ENDGAME_REVEAL_OPTIONS / villager_ai.ENDGAME_BLUFF_
# OPTIONS all use: stay_silent/stay_hidden, claim_village, claim_wolf,
# claim_madman.
_ENDGAME_CLAIM_LABELS_JP = ["黙っている", "村人CO", "狼CO", "狂人CO"]


def _endgame_matrix_spec(option_labels_jp):
    """
    option_labels_jp must list all N option labels IN THE SAME ORDER as
    the module's OPTIONS list (wolf_ai.ENDGAME_DECLARE_OPTIONS /
    madman_ai.ENDGAME_REVEAL_OPTIONS / villager_ai.ENDGAME_BLUFF_OPTIONS),
    since _matrix_value_index indexes into the weight vector by that
    order. All three now share the same 4-option order: stay silent/
    hidden, claim_village, claim_wolf, claim_madman.
    """
    option_headers = list(option_labels_jp)
    row_labels = _endgame_context_labels()
    return option_headers, row_labels, "context_major"


# snapshot key -> () -> (option_headers, row_labels, layout)
_MATRIX_SPECS = {
    "wolf_co": _wolf_co_matrix_spec,
    "madman_co": _wolf_co_matrix_spec,  # 狂人COはwolf_ai._co_feature_listをそのまま再利用しているので同じ形
    "hunter_co": _hunter_co_matrix_spec,
    "villager_endgame_bluff": lambda: _endgame_matrix_spec(_ENDGAME_CLAIM_LABELS_JP),
    "madman_endgame_reveal": lambda: _endgame_matrix_spec(_ENDGAME_CLAIM_LABELS_JP),
    "wolf_endgame_declare": lambda: _endgame_matrix_spec(_ENDGAME_CLAIM_LABELS_JP),
}


# ============================================================
# 学習の前後比較レポート(日本語)
#
# train() の実行前後で全方策の重みをスナップショットし、どの特徴量が
# どちらの方向にどれだけ動いたかを日本語で報告する。「重みが動いた」
# だけでなく変化幅の大小も一緒に示すことで、まだ揺れているだけの
# 特徴量と、明確に定着してきた特徴量を区別しやすくしている。
# ============================================================

def _snapshot_all_weights():
    return {
        "village_execution": ("村AI: 処刑投票(執行方策)の重み",
                               _village_execution_labels(),
                               list(village_ai.execution_policy.weights)),
        "village_guard_instruction": ("村AI: 護衛指示(狩人2人COのとき)の重み",
                                       _village_guard_instruction_labels(),
                                       list(village_ai.guard_instruction_policy.weights)),
        "wolf_co": ("狼AI: 朝一CO(何を騙るか)の重み",
                    _wolf_co_labels(),
                    list(wolf_ai.co_policy.weights)),
        "wolf_attack": ("狼AI: 襲撃先選択の重み",
                        WOLF_ATTACK_LABELS,
                        list(wolf_ai.attack_policy.weights)),
        "wolf_vote": ("狼AI: 投票先選択の重み",
                      WOLF_VOTE_LABELS,
                      list(wolf_ai.vote_policy.weights)),
        "wolf_fake_declaration": ("狼AI: 偽狩人の護衛先偽装の重み",
                                  WOLF_FAKE_DECLARATION_LABELS,
                                  list(wolf_ai.fake_declaration_policy.weights)),
        "wolf_fake_report": ("狼AI: 偽霊能の報告偽装の重み",
                             WOLF_FAKE_REPORT_LABELS,
                             list(wolf_ai.fake_report_policy.weights)),
        "wolf_fake_seer_report": ("狼AI: 偽占い師の報告偽装の重み",
                                  WOLF_FAKE_SEER_REPORT_LABELS,
                                  list(wolf_ai.fake_seer_report_policy.weights)),
        "wolf_accuse_target": ("狼AI: 対抗占い師の誰を黒判定にするかの重み",
                               ACCUSE_TARGET_LABELS,
                               list(wolf_ai.accuse_target_policy.weights)),
        "wolf_endgame_declare": ("狼AI: 最終盤の「正体を明かす」偽装の重み",
                                 _endgame_decision_labels(_ENDGAME_CLAIM_LABELS_JP),
                                 list(wolf_ai.endgame_declare_policy.weights)),
        "wolf_day1_joint_co": ("狼AI: 初日CO(2人の共同プラン)の重み",
                               [
                                   f"{_wolf_co_option_jp(a)} + {_wolf_co_option_jp(b)}"
                                   for a, b in wolf_ai.DAY1_JOINT_CO_OPTIONS
                               ],
                               list(wolf_ai.day1_joint_co_policy.weights)),
        "medium_nomination": ("霊能AI: 初日指定の重み",
                              MEDIUM_NOMINATION_LABELS,
                              list(medium_ai.nomination_policy.weights)),
        "medium_nomination_target": ("霊能AI: 指定する相手の選び方の重み"
                                      "(村AIの処刑投票と同じ特徴量を共有)",
                                      _village_execution_labels(),
                                      list(medium_ai.nomination_target_policy.weights)),
        "hunter_co": ("狩人AI: CO(告白)タイミングの重み",
                      HUNTER_CO_LABELS,
                      list(hunter_ai.co_policy.weights)),
        "hunter_guard": ("狩人AI: 護衛先選択の重み",
                         HUNTER_GUARD_LABELS,
                         list(hunter_ai.guard_policy.weights)),
        "seer_investigate": ("占い師AI: 占う相手の選び方の重み",
                             SEER_INVESTIGATE_LABELS,
                             list(seer_ai.investigate_policy.weights)),
        "villager_endgame_bluff": ("村人AI: 最終盤の「村騙り」の重み",
                                   _endgame_decision_labels(_ENDGAME_CLAIM_LABELS_JP),
                                   list(villager_ai.endgame_bluff_policy.weights)),
        "madman_vote": ("狂人AI: 投票先選択の重み(村AIと同じ特徴量を共有)",
                        _village_execution_labels(),
                        list(madman_ai.vote_policy.weights)),
        "madman_co": ("狂人AI: 朝一CO(何を騙るか)の重み(狼AIと同じ形)",
                      _wolf_co_labels(),
                      list(madman_ai.co_policy.weights)),
        "madman_fake_report": ("狂人AI: 偽霊能の報告偽装の重み",
                               WOLF_FAKE_REPORT_LABELS,
                               list(madman_ai.fake_report_policy.weights)),
        "madman_fake_seer_report": ("狂人AI: 偽占い師の報告偽装の重み",
                                    madman_ai.MADMAN_FAKE_SEER_REPORT_OPTIONS,
                                    list(madman_ai.fake_seer_report_policy.weights)),
        "madman_accuse_target": ("狂人AI: 対抗占い師の誰を黒判定にするかの重み",
                                 ACCUSE_TARGET_LABELS,
                                 list(madman_ai.accuse_target_policy.weights)),
        "madman_endgame_reveal": ("狂人AI: 最終盤の「捨てCO」の重み",
                                  _endgame_decision_labels(_ENDGAME_CLAIM_LABELS_JP),
                                  list(madman_ai.endgame_reveal_policy.weights)),
        "value_function": ("価値関数(状況の有利/不利評価)の重み",
                           VALUE_FUNCTION_LABELS,
                           list(value_function.value_weights)),
    }


def _describe_change(delta):
    if delta > 0.5:
        return "大きく上昇"
    if delta > 0.05:
        return "上昇"
    if delta < -0.5:
        return "大きく低下"
    if delta < -0.05:
        return "低下"
    return "微増" if delta > 0 else "微減" if delta < 0 else "変化なし"


WOLF_SNAPSHOT_KEYS = [
    "wolf_day1_joint_co", "wolf_co", "wolf_attack", "wolf_vote",
    "wolf_fake_declaration", "wolf_fake_report", "wolf_fake_seer_report",
    "wolf_accuse_target", "wolf_endgame_declare",
]

# 狂人AI固有の方策(狼AIの機能を流用しているものも含む)。以前はこれらの
# キーが _snapshot_all_weights() に存在するだけでどのレポートファイルにも
# 書き出されておらず、変化を確認する手段がなかった -- write_madman_change_
# report で他のAI同様にテキストファイル化する。
MADMAN_SNAPSHOT_KEYS = [
    "madman_co", "madman_vote", "madman_fake_report",
    "madman_fake_seer_report", "madman_accuse_target", "madman_endgame_reveal",
]

VILLAGE_SNAPSHOT_KEYS = [
    "village_execution", "village_guard_instruction",
]

HUNTER_SNAPSHOT_KEYS = [
    "hunter_co", "hunter_guard",
]

SEER_SNAPSHOT_KEYS = [
    "seer_investigate",
]

# medium_co は現状 main.py 側で未使用(Day1強制COに置き換え済み)のため
# 対象外。実際に使われている nomination_policy だけを対象にする。
MEDIUM_SNAPSHOT_KEYS = [
    "medium_nomination",
    "medium_nomination_target",
]


def _build_change_report_text(before, after, threshold=0.02, keys=None,
                               header="今回の学習で何が変わったか(前後比較)",
                               show_all_weights=True):
    """
    `before` / `after` は _snapshot_all_weights() の戻り値。

    まず、しきい値(デフォルト0.02)以上動いた特徴量だけを、変化幅の
    大きい順に日本語でまとめる(表示にもファイル保存にも同じテキストを
    使い回せるように、print せずに文字列として返す)。

    show_all_weights=True(デフォルト)の場合、その後ろに各方策が
    持つ特徴量「全件」の現在値(重み)を一覧として付け加える。これは
    このAIの判断根拠になっている情報のすべて(線形方策の重みベクトル
    全体 -- 変化がゼロだった特徴量も含む)を漏れなく確認できるように
    するためのもの。今回動かなかった項目も、既に学習済みで判断に
    使われ続けている重要な特徴量である場合があるため。

    keys : 対象を絞りたい場合、_snapshot_all_weights() のキーの
        リスト(例: WOLF_SNAPSHOT_KEYS)。None なら全方策が対象。
    """
    lines = []
    lines.append("=" * 78)
    lines.append(header)
    lines.append("=" * 78)

    target_keys = keys if keys is not None else list(after.keys())

    lines.append("\n【1. 今回の学習で動いた特徴量(変化の大きい順、"
                 "option×context型はマトリクス形式)】")
    any_change = False
    for key in target_keys:
        if key not in after:
            continue
        title, labels, after_weights = after[key]
        before_weights = before[key][2]

        if key == "wolf_day1_joint_co":
            deltas = [a - b for a, b in zip(after_weights, before_weights)]
            if all(abs(d) < threshold for d in deltas):
                continue
            any_change = True
            base_options_jp = [_wolf_co_option_jp(o) for o in wolf_ai.CO_OPTIONS]
            lookup = _wolf_day1_joint_co_weight_lookup(deltas)
            lines.append(f"\n■ {title}")
            lines.append("  [変化量 Δ = after - before] (対称行列: 狼A/Bの区別はない)  "
                          "(|Δ|≥0.5→** / |Δ|≥0.05→* / それ未満は印なし)")
            lines += [
                "  " + row for row in
                _render_symmetric_pair_matrix(base_options_jp, lookup, _format_delta_cell)
            ]
            continue

        if key in _MATRIX_SPECS:
            option_headers, row_labels, layout = _MATRIX_SPECS[key]()
            n_opt, n_ctx = len(option_headers), len(row_labels)
            deltas = [a - b for a, b in zip(after_weights, before_weights)]
            if all(abs(d) < threshold for d in deltas):
                continue
            any_change = True

            def value_fn(ctx_i, opt_i, _deltas=deltas, _layout=layout,
                         _n_opt=n_opt, _n_ctx=n_ctx):
                return _deltas[_matrix_value_index(_layout, _n_opt, _n_ctx, opt_i, ctx_i)]

            lines.append(f"\n■ {title}")
            lines.append("  [変化量 Δ = after - before]  "
                          "(|Δ|≥0.5→** / |Δ|≥0.05→* / それ未満は印なし)")
            lines += [
                "  " + row for row in
                _render_matrix(option_headers, row_labels, value_fn, _format_delta_cell)
            ]
            continue

        rows = []
        for label, b, a in zip(labels, before_weights, after_weights):
            delta = a - b
            if abs(delta) >= threshold:
                rows.append((abs(delta), label, b, a, delta))
        if not rows:
            continue
        any_change = True
        rows.sort(key=lambda r: r[0], reverse=True)
        lines.append(f"\n■ {title}")
        # _pad_left (not plain str.ljust/{:<N}) so the ":" lines up in
        # the terminal regardless of how many full-width (2-column)
        # Japanese characters are in each label -- see _display_width's
        # docstring. Width is per-SECTION (this policy's own labels
        # only), not a fixed 45, so a short-labeled policy doesn't get
        # padded out to match some other policy's longest label.
        label_width = max(_display_width(label) for _, label, _, _, _ in rows)
        for _, label, b, a, delta in rows:
            arrow = "↑" if delta > 0 else "↓"
            lines.append(f"  {_pad_left(label, label_width)}: {b:+.3f} → {a:+.3f}  "
                          f"{arrow}{abs(delta):.3f}  ({_describe_change(delta)})")

    if not any_change:
        lines.append(f"\n(しきい値 {threshold} を超える変化のあった特徴量はありませんでした)")

    if show_all_weights:
        lines.append("\n" + "=" * 78)
        lines.append("【2. 現時点の重み全件(このAIが判断に使っている情報のすべて、"
                      "option×context型はマトリクス形式)】")
        lines.append("=" * 78)
        lines.append("※ 上の1では変化が小さくて表示されなかった特徴量も、"
                      "ここには全件そのまま載っています。値が0に近いものは"
                      "「まだ判断材料として使われていない/重視されていない」"
                      "ことを、絶対値が大きいものは「強く判断に影響している」"
                      "ことを意味します。")
        for key in target_keys:
            if key not in after:
                continue
            title, labels, after_weights = after[key]

            if key == "wolf_day1_joint_co":
                base_options_jp = [_wolf_co_option_jp(o) for o in wolf_ai.CO_OPTIONS]
                lookup = _wolf_day1_joint_co_weight_lookup(after_weights)
                lines.append(f"\n■ {title} (対称行列: 狼A/Bの区別はない)")
                lines += [
                    "  " + row for row in
                    _render_symmetric_pair_matrix(base_options_jp, lookup, _format_weight_cell)
                ]
                continue

            if key in _MATRIX_SPECS:
                option_headers, row_labels, layout = _MATRIX_SPECS[key]()
                n_opt, n_ctx = len(option_headers), len(row_labels)

                def value_fn(ctx_i, opt_i, _w=after_weights, _layout=layout,
                             _n_opt=n_opt, _n_ctx=n_ctx):
                    return _w[_matrix_value_index(_layout, _n_opt, _n_ctx, opt_i, ctx_i)]

                lines.append(f"\n■ {title}")
                lines += [
                    "  " + row for row in
                    _render_matrix(option_headers, row_labels, value_fn, _format_weight_cell)
                ]
                continue

            rows = sorted(
                zip(labels, after_weights),
                key=lambda lw: abs(lw[1]),
                reverse=True,
            )
            lines.append(f"\n■ {title}(全{len(rows)}項目)")
            label_width = max(_display_width(label) for label, _ in rows)
            for label, w in rows:
                lines.append(f"  {_pad_left(label, label_width)}: {w:+.3f}")

    return "\n".join(lines)


def print_change_report(before, after, threshold=0.02):
    print()
    print(_build_change_report_text(before, after, threshold))


WOLF_REPORT_PATH = "wolf_ai_change_report.txt"


def write_wolf_change_report(before, after, threshold=0.02, path=WOLF_REPORT_PATH, verbose=False, announce=True):
    """
    狼(werewolf) AI に関係する方策(朝一CO・襲撃先・投票先・偽装系)
    だけに絞った前後比較レポートを、日本語のテキストファイルとして
    書き出す。呼ぶたびに上書きされる(最新の学習1回分の差分)。

    verbose=False (train() のデフォルト、2026-10変更): ファイルへの
    書き出しは常に行うが、コンソールへの全文表示はしない -- 6つの
    change_report全部をコンソールに流すと1回のtrain()実行の出力が
    長くなりすぎてターミナルのスクロールバックから溢れてしまう
    (チャットで報告された問題)。中身を見たい時は書き出された
    {path} を直接開くか、verbose=True で呼び直す。
    """
    text = _build_change_report_text(
        before, after, threshold, keys=WOLF_SNAPSHOT_KEYS,
        header="狼AI(werewolf_ai)の学習前後比較",
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    if verbose:
        print()
        print(text)
    if announce:
        print(f"\n(このレポートは {path} に保存されました)")


VILLAGE_REPORT_PATH = "village_ai_change_report.txt"


def write_village_change_report(before, after, threshold=0.02, path=VILLAGE_REPORT_PATH, verbose=True, announce=True):
    """
    村側(処刑投票・護衛指示)に絞った前後比較レポートを、日本語の
    テキストファイルとして書き出す。write_wolf_change_report の村版。

    verbose=True がデフォルト(他5つのwrite_*_change_reportとは逆):
    村側の処刑投票の重み("隠れ層")は盤面別勝率と直結する、今
    一番追いたい数字なので、これだけはtrain()からもコンソールに
    表示されたままにしてある。他の5つ(狼/狩人/霊能/占い/狂人)は
    verbose=Falseがデフォルトでファイルのみの出力(2026-10、
    ターミナルの出力が長すぎる問題への対応)。
    """
    text = _build_change_report_text(
        before, after, threshold, keys=VILLAGE_SNAPSHOT_KEYS,
        header="村AI(village_ai)の学習前後比較",
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    if verbose:
        print()
        print(text)
    if announce:
        print(f"\n(このレポートは {path} に保存されました)")


HUNTER_REPORT_PATH = "hunter_ai_change_report.txt"


def write_hunter_change_report(before, after, threshold=0.02, path=HUNTER_REPORT_PATH, verbose=False, announce=True):
    """
    狩人AI(朝一CO・護衛先選択)に絞った前後比較レポートを、日本語の
    テキストファイルとして書き出す。verbose=Falseがデフォルト --
    write_wolf_change_report のverboseコメント参照。
    """
    text = _build_change_report_text(
        before, after, threshold, keys=HUNTER_SNAPSHOT_KEYS,
        header="狩人AI(hunter_ai)の学習前後比較",
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    if verbose:
        print()
        print(text)
    if announce:
        print(f"\n(このレポートは {path} に保存されました)")


MEDIUM_REPORT_PATH = "medium_ai_change_report.txt"


def write_medium_change_report(before, after, threshold=0.02, path=MEDIUM_REPORT_PATH, verbose=False, announce=True):
    """
    霊能AI(灰指定するかどうか)に絞った前後比較レポートを、日本語の
    テキストファイルとして書き出す。verbose=Falseがデフォルト --
    write_wolf_change_report のverboseコメント参照。
    """
    text = _build_change_report_text(
        before, after, threshold, keys=MEDIUM_SNAPSHOT_KEYS,
        header="霊能AI(medium_ai)の学習前後比較",
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    if verbose:
        print()
        print(text)
    if announce:
        print(f"\n(このレポートは {path} に保存されました)")


SEER_REPORT_PATH = "seer_ai_change_report.txt"


def write_seer_change_report(before, after, threshold=0.02, path=SEER_REPORT_PATH, verbose=False, announce=True):
    """
    占い師AI(CO判断・占う相手選択)に絞った前後比較レポートを、
    日本語のテキストファイルとして書き出す。verbose=Falseがデフォルト
    -- write_wolf_change_report のverboseコメント参照。
    """
    text = _build_change_report_text(
        before, after, threshold, keys=SEER_SNAPSHOT_KEYS,
        header="占い師AI(seer_ai)の学習前後比較",
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    if verbose:
        print()
        print(text)
    if announce:
        print(f"\n(このレポートは {path} に保存されました)")


MADMAN_REPORT_PATH = "madman_ai_change_report.txt"


def write_madman_change_report(before, after, threshold=0.02, path=MADMAN_REPORT_PATH, verbose=False, announce=True):
    """
    狂人(madman) AI に絞った前後比較レポートを、日本語のテキスト
    ファイルとして書き出す。write_wolf_change_report の狂人版。
    verbose=Falseがデフォルト -- write_wolf_change_report の
    verboseコメント参照。
    """
    text = _build_change_report_text(
        before, after, threshold, keys=MADMAN_SNAPSHOT_KEYS,
        header="狂人AI(madman_ai)の学習前後比較",
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    if verbose:
        print()
        print(text)
    if announce:
        print(f"\n(このレポートは {path} に保存されました)")


def list_village_weights():
    print()
    print("=" * 78)
    print("VILLAGE AI -- execution weights")
    print("=" * 78)
    labels = _village_execution_labels()
    label_width = max(_display_width(label) for label in labels)
    for label, w in zip(labels, village_ai.execution_policy.weights):
        print(f"  {_pad_left(label, label_width)}: {w:+.3f}")

    print()
    print("guard-instruction strategy weights "
          "[protect_medium, both_random]:")
    print(f"  {[round(w, 3) for w in village_ai.guard_instruction_policy.weights]}")

    print()
    print("=" * 78)
    print("WOLF -- vote weights "
          "[bias, is_teammate, is_Ps_claimant, is_Hn_claimant, is_confirmed_villager, day]")
    print("=" * 78)
    print(f"  {[round(w, 3) for w in wolf_ai.vote_policy.weights]}")


def list_co_policies():
    print()
    print("=" * 78)
    print("MEDIUM -- P(reveal) by day")
    print("=" * 78)
    for day in range(1, 5):
        m_feat = medium_ai._co_feature_list({"day": day})
        m_probs = policy_utils.softmax(
            [medium_ai.co_policy.score(f) for f in m_feat]
        )
        print(f"day {day}: P(reveal)={m_probs[0]:.3f}")

    print()
    print("=" * 78)
    print("HUNTER -- P(reveal) by (guard successes so far, gray count)")
    print("=" * 78)
    print(f"{'guards':>7} {'gray#':>6} {'P(reveal)':>10}")
    rows = []
    for gc in range(3):
        for gray in range(6):
            h_feat = hunter_ai._co_feature_list(
                {"guard_success_count": gc, "gray_count": gray}
            )
            h_probs = policy_utils.softmax(
                [hunter_ai.co_policy.score(f) for f in h_feat]
            )
            rows.append((gc, gray, h_probs[0]))
    for gc, gray, p in rows:
        print(f"{gc:7d} {gray:6d} {p:10.3f}")

    print()
    print("=" * 78)
    print("WOLF -- P(claim), by what's already publicly known")
    print("=" * 78)
    for medium_revealed in [False, True]:
        for hunter_revealed in [False, True]:
            state = {
                "day": 2,
                "co_history": (["Ps"] if medium_revealed else [None]) +
                              (["Hn"] if hunter_revealed else [None]),
            }
            feat = wolf_ai._co_feature_list(state)
            probs = policy_utils.softmax(
                [wolf_ai.co_policy.score(f) for f in feat]
            )
            probs_str = "  ".join(
                f"P({opt})={p:.3f}" for opt, p in zip(wolf_ai.CO_OPTIONS, probs)
            )
            print(f"medium_revealed={str(medium_revealed):<5} "
                  f"hunter_revealed={str(hunter_revealed):<5} -> {probs_str}")


def full_report():
    list_co_policies()
    list_village_weights()


if __name__ == "__main__":
    train()
    # full_report()  # 盤面ごとの一覧表示は、必要になったらコメントを外してください
