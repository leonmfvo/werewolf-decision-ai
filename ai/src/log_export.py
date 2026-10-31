"""
log_export.py -- turn the text log printed by `python3 main.py` into a
plain dict / JSON structure that can be loaded with json.load().

Two entry points:

    parse_log(text)          -> dict   (one game's log text -> dict)
    generate_games(n, seed)  -> list   (play n fresh games with the CURRENT
                                        trained weights and parse each one)

CLI:
    python3 log_export.py 200 --seed 1 --out games.json

Design notes
------------
* The parser works on the PRINTED log, not on main.py's internals, so it keeps
  working for any log produced by `python3 main.py` (or the spike logs saved
  during training) and main.py itself is never modified.
* Every line the parser does not recognise is kept in game["unparsed"] (and
  the raw text in game["log"]) -- nothing is silently dropped. A healthy run
  has unparsed == [] for every game; check_unparsed() reports otherwise.
* Players are keyed "P1".."P9". Roles use English keys:
  wolf / madman / seer / medium / hunter / villager.
"""

import argparse
import contextlib
import io
import json
import random
import re
import sys

ROLE_KEY = {"狼": "wolf", "狂": "madman", "占": "seer", "霊": "medium",
            "狩": "hunter", "村": "villager"}
CLAIM_KEY = {"占": "seer", "霊": "medium", "狩": "hunter"}
ENDGAME_KEY = {"狼": "wolf", "狂人": "madman", "村人": "villager"}
VERDICT = {"白": "white", "黒": "black"}

_P = r"(P\d+)"

RE_HEADER = re.compile(r"^={10} (\d+)日目 (昼|夜) ={10}$")
RE_ROLES = re.compile(r"^配役: (.*)$")
RE_ROLE_ITEM = re.compile(r"(P\d+)\((.)\)")
RE_CO = re.compile(rf"^{_P}【(占|霊|狩)CO】$")
RE_BOARD = re.compile(r"^盤面\(占-霊-狩\): (.*)$")
RE_SEER = re.compile(rf"^【占い結果】{_P}: {_P}は(白|黒)でした$")
RE_MEDIUM = re.compile(rf"^【霊能結果】{_P}: {_P}は(白|黒)でした$")
RE_MEDIUM_LATE = re.compile(
    rf"^\s*→ \(遡って確定\) (\d+)日目の霊能結果: {_P}: {_P}は(白|黒)でした$")
RE_NOMINATE = re.compile(rf"^{_P}\(霊CO\)が{_P}を指定$")
RE_EMERGENCY = re.compile(rf"^{_P} が(?:処刑対象に指名され|対抗して)、?(?:回避のため)?緊急CO$")
RE_NOM_CHANGE = re.compile(rf"^\s*→ {_P}\(指定者\)が指定を変更: {_P} → {_P}$")
RE_GUARD_INSTR = re.compile(r"^【護衛指示】(.*)$")
RE_GUARD_INSTR_ITEM = re.compile(rf"{_P}\(狩CO\) → {_P} を護衛")
RE_VOTE = re.compile(r"^【投票】(.*)$")
RE_VOTE_ITEM = re.compile(rf"{_P}→{_P}")
RE_TIE = re.compile(r"^\s*→ (\d+)票で同数\((.*)\) → ランダムで" + _P + "に決定$")
RE_EXEC = re.compile(rf"^処刑先: {_P} \((.+)\)$")
RE_VILLAGE_LABEL = re.compile(r"^村側の最善進行>>(.*)$")
RE_WOLF_LABEL = re.compile(r"^狼側の最善進行>>(.*)$")
RE_GUARD_TRUE = re.compile(rf"^狩人の護衛先: {_P}$")
RE_SEER_NIGHT = re.compile(rf"^{_P}\(占い\)が{_P}を占った$")
RE_ATTACK = re.compile(rf"^人狼の襲撃先: {_P}$")
RE_ATTACKED = re.compile(rf"^→ {_P} が襲撃された$")
RE_GUARD_OK = re.compile(r"^→ 護衛成功!襲撃は失敗した$")
RE_PROVEN_FAKE = re.compile(rf"^{_P} の護衛先が死亡 → {_P} は狩人偽物と判明!$")
RE_PROVEN_REAL = re.compile(rf"^{_P} の護衛先が生存 → {_P} は本物の狩人と判明!$")
RE_ALIVE = re.compile(r"^生存\((\d+)\): (.*) / 死亡: (.*)$")
RE_ENDGAME = re.compile(rf"^{_P}が正体を明かした:【(狼|狂人|村人)CO】$")
RE_FALSE_VOUCH = re.compile(
    rf"^\(嘘\)「{_P} を護衛しました、白です」と{_P}が主張\(※本当は偽り\)$")
RE_TRUE_VOUCH = re.compile(
    rf"^「{_P} を護衛しました、白です」と{_P}が主張$")
RE_REVEAL_GUARDS = re.compile(
    rf"^\s*→ (?:これまでの護衛先を公表|(?:{_P}) が護衛先を公表): ((?:P\d+(?:、)?)+) は確定で白!$")
RE_WINNER = re.compile(r"^勝者: (村人陣営|人狼陣営)$")


def _new_day(day):
    return {
        "day": day,
        "day_phase": {
            "co": [], "seer_reports": [], "medium_reports": [],
            "late_medium_reports": [], "nominations": [], "emergency_co": [], "nomination_changes": [],
            "guard_instructions": [], "guard_vouches": [],
            "endgame_claims": [], "votes": {}, "vote_tie": None,
            "execution": None, "village_line": None,
        },
        "night_phase": {
            "hunter_guard": None, "seer_investigation": None,
            "attack": None, "wolf_line": None, "attack_result": None,
            "guard_verification": [], "alive_after": None, "dead_after": None,
        },
    }


def parse_log(text):
    game = {
        "roles": {}, "winner": None, "board_detail": None,
        "days": [], "unparsed": [], "log": text,
    }
    cur = None      # current day dict
    phase = None    # "day_phase" | "night_phase"

    for raw in text.splitlines():
        line = raw.rstrip()
        s = line.strip()
        if not s or set(s) <= {"="} or "人狼ゲーム" in s:
            continue

        m = RE_ROLES.match(line)
        if m:
            for pid, r in RE_ROLE_ITEM.findall(m.group(1)):
                game["roles"][pid] = ROLE_KEY[r]
            continue

        m = RE_HEADER.match(line)
        if m:
            day = int(m.group(1))
            if not game["days"] or game["days"][-1]["day"] != day:
                cur = _new_day(day)
                game["days"].append(cur)
            cur = game["days"][-1]
            phase = "day_phase" if m.group(2) == "昼" else "night_phase"
            continue

        m = RE_WINNER.match(line)
        if m:
            game["winner"] = "village" if m.group(1) == "村人陣営" else "werewolf"
            continue

        if cur is None:
            game["unparsed"].append(line)
            continue

        d = cur["day_phase"]
        n = cur["night_phase"]

        if (m := RE_CO.match(line)):
            d["co"].append({"player": m.group(1), "claim": CLAIM_KEY[m.group(2)]})
        elif (m := RE_BOARD.match(line)):
            game["board_detail"] = m.group(1)
        elif (m := RE_SEER.match(line)):
            d["seer_reports"].append({"claimant": m.group(1), "target": m.group(2),
                                      "verdict": VERDICT[m.group(3)]})
        elif (m := RE_MEDIUM.match(line)):
            d["medium_reports"].append({"claimant": m.group(1), "target": m.group(2),
                                        "verdict": VERDICT[m.group(3)]})
        elif (m := RE_MEDIUM_LATE.match(line)):
            d["late_medium_reports"].append({
                "about_day": int(m.group(1)), "claimant": m.group(2),
                "target": m.group(3), "verdict": VERDICT[m.group(4)]})
        elif (m := RE_NOMINATE.match(line)):
            d["nominations"].append({"medium_claimant": m.group(1), "target": m.group(2)})
        elif (m := RE_EMERGENCY.match(line)):
            d["emergency_co"].append(m.group(1))
        elif (m := RE_NOM_CHANGE.match(line)):
            d["nomination_changes"].append({"medium_claimant": m.group(1), "from": m.group(2), "to": m.group(3)})
        elif (m := RE_GUARD_INSTR.match(line)):
            for c, t in RE_GUARD_INSTR_ITEM.findall(m.group(1)):
                d["guard_instructions"].append({"claimant": c, "target": t})
        elif (m := RE_VOTE.match(line)):
            d["votes"] = {a: b for a, b in RE_VOTE_ITEM.findall(m.group(1))}
        elif (m := RE_TIE.match(line)):
            d["vote_tie"] = {"votes": int(m.group(1)),
                             "tied": re.findall(r"P\d+", m.group(2)),
                             "random_pick": m.group(3)}
        elif (m := RE_EXEC.match(line)):
            d["execution"] = {"target": m.group(1),
                              "revealed_role": ROLE_KEY.get(m.group(2), m.group(2))}
        elif (m := RE_VILLAGE_LABEL.match(line)):
            d["village_line"] = m.group(1)
        elif (m := RE_ENDGAME.match(line)):
            d["endgame_claims"].append({"player": m.group(1),
                                        "claim": ENDGAME_KEY[m.group(2)]})
        elif (m := RE_FALSE_VOUCH.match(line)):
            d["guard_vouches"].append({"claimant": m.group(2), "target": m.group(1),
                                       "is_lie": True})
        elif (m := RE_TRUE_VOUCH.match(line)):
            d["guard_vouches"].append({"claimant": m.group(2), "target": m.group(1),
                                       "is_lie": False})
        elif (m := RE_REVEAL_GUARDS.match(line)):
            claimant = m.group(1) or (d["co"][-1]["player"] if d["co"] else None)
            for t in re.findall(r"P\d+", m.group(2)):
                d["guard_vouches"].append({"claimant": claimant, "target": t,
                                           "is_lie": False, "kind": "history_reveal"})
        elif (m := RE_GUARD_TRUE.match(line)):
            n["hunter_guard"] = m.group(1)
        elif (m := RE_SEER_NIGHT.match(line)):
            n["seer_investigation"] = {"seer": m.group(1), "target": m.group(2)}
        elif (m := RE_ATTACK.match(line)):
            n["attack"] = m.group(1)
        elif (m := RE_WOLF_LABEL.match(line)):
            n["wolf_line"] = m.group(1)
        elif (m := RE_ATTACKED.match(line)):
            n["attack_result"] = "killed"
        elif RE_GUARD_OK.match(line):
            n["attack_result"] = "guarded"
        elif (m := RE_PROVEN_FAKE.match(line)):
            n["guard_verification"].append({"claimant": m.group(2), "result": "fake"})
        elif (m := RE_PROVEN_REAL.match(line)):
            n["guard_verification"].append({"claimant": m.group(2), "result": "real"})
        elif (m := RE_ALIVE.match(line)):
            n["alive_after"] = m.group(2).split()
            n["dead_after"] = m.group(3).split()
        else:
            game["unparsed"].append(line)

    game["features"] = extract_features(game)
    return game


def extract_features(game):
    """Flat, dictionary-style features derived from the parsed game."""
    roles = game["roles"]
    days = game["days"]
    f = {}

    f["winner"] = game["winner"]
    f["village_win"] = game["winner"] == "village"
    f["num_days"] = len(days)
    f["num_players"] = len(roles)
    f["absent_roles"] = [r for r in ("seer", "medium", "hunter", "madman")
                         if r not in roles.values()]

    # Day-1 claim counts (seer-medium-hunter), same convention as the
    # "2-2-2" board notation used in training.
    day1_co = days[0]["day_phase"]["co"] if days else []
    counts = {k: sum(1 for c in day1_co if c["claim"] == k)
              for k in ("seer", "medium", "hunter")}
    f["claim_counts"] = counts
    f["board_config"] = f"{counts['seer']}-{counts['medium']}-{counts['hunter']}"
    f["board_detail"] = game["board_detail"]
    f["claims_by_truth"] = {
        k: {"genuine": sum(1 for c in day1_co if c["claim"] == k
                           and roles.get(c["player"]) == k),
            "fake": sum(1 for c in day1_co if c["claim"] == k
                        and roles.get(c["player"]) != k)}
        for k in ("seer", "medium", "hunter")
    }
    f["fake_claimant_roles"] = sorted({
        roles[c["player"]] for c in day1_co if roles.get(c["player"]) != c["claim"]
    })

    executions = [d["day_phase"]["execution"] for d in days
                  if d["day_phase"]["execution"]]
    f["executed_roles"] = [e["revealed_role"] for e in executions]
    f["wolves_executed"] = sum(1 for e in executions if e["revealed_role"] == "wolf")
    f["first_execution_role"] = f["executed_roles"][0] if executions else None
    f["village_lines"] = [d["day_phase"]["village_line"] for d in days
                          if d["day_phase"]["village_line"]]

    attacks = [d["night_phase"] for d in days if d["night_phase"]["attack"]]
    f["attacked_roles"] = [roles.get(a["attack"]) for a in attacks]
    f["wolf_lines"] = [a["wolf_line"] for a in attacks if a["wolf_line"]]
    f["guard_success"] = any(a["attack_result"] == "guarded" for a in attacks)
    f["night_kills"] = sum(1 for a in attacks if a["attack_result"] == "killed")

    ver = [v for d in days for v in d["night_phase"]["guard_verification"]]
    f["hunter_verified_fake"] = any(v["result"] == "fake" for v in ver)
    f["hunter_verified_real"] = any(v["result"] == "real" for v in ver)
    f["guard_instruction_days"] = [d["day"] for d in days
                                   if d["day_phase"]["guard_instructions"]]
    f["vote_ties"] = sum(1 for d in days if d["day_phase"]["vote_tie"])
    f["endgame_claims"] = [c for d in days for c in d["day_phase"]["endgame_claims"]]

    # Per-day vote structure (who voted for a wolf), useful as a feature.
    wolf_votes = []
    for d in days:
        votes = d["day_phase"]["votes"]
        if votes:
            wolf_votes.append(sum(1 for t in votes.values() if roles.get(t) == "wolf"))
    f["votes_on_wolves_per_day"] = wolf_votes
    return f


def check_unparsed(games):
    """Return {line: count} of lines the parser did not recognise."""
    bad = {}
    for g in games:
        for ln in g["unparsed"]:
            bad[ln] = bad.get(ln, 0) + 1
    return bad


def generate_games(n, seed=None, quiet_load=True, start_id=1):
    """Play n games with the CURRENT trained weights; return parsed dicts.

    Weights are loaded read-only (never saved), so this cannot disturb a
    training run's trained_weights.pkl.
    """
    import persistence
    import main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf if quiet_load else sys.stdout):
        persistence.load_weights()
    if seed is not None:
        random.seed(seed)

    # Villagers and wolves are trained together in every self-play epoch,
    # so both sides have seen the same cumulative epoch count. (A policy
    # whose feature dimension changed was reset when it was loaded; that
    # per-policy reset point is not recorded in the weights file.)
    training = {
        "village_epochs": persistence.total_epochs,
        "wolf_epochs": persistence.total_epochs,
        "total_games_trained": persistence.total_games,
    }
    games = []
    for i in range(n):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main.run_game(verbose=True)
        g = parse_log(out.getvalue())
        g["game_id"] = start_id + i
        g["training"] = dict(training)
        games.append(g)
    return games


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=100)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="games.json")
    ap.add_argument("--start-id", type=int, default=1, help="first game_id (for continuing numbering across shard files)")
    ap.add_argument("--no-raw-log", action="store_true",
                    help="drop the raw text log from each game (smaller file)")
    args = ap.parse_args()

    games = generate_games(args.n, args.seed, start_id=args.start_id)
    bad = check_unparsed(games)
    if bad:
        print("UNPARSED LINES:")
        for ln, c in sorted(bad.items(), key=lambda x: -x[1]):
            print(f"  {c:4d}  {ln}")
    if args.no_raw_log:
        for g in games:
            g.pop("log", None)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(games, fh, ensure_ascii=False)
    print(f"wrote {len(games)} games -> {args.out}  (unparsed kinds: {len(bad)})")


if __name__ == "__main__":
    main_cli()
