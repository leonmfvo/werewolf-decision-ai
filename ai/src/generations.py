"""
generations.py -- start training from zero and watch strength per generation.

Commands (run from this directory):

  Add  --tag NAME  to keep a separate run (own weights file and generations
  folder, e.g. --tag lr003), and  train --lr 0.003  to change the learning
  rate (default 0.03).

  Add  --mode 9std  (before or after the command) to train the fixed
  standard-9 setup (no role absence). It has its own weights file
  (trained_weights_9std.pkl) and folder (generations_9std/).

  python3 generations.py reset            back up current weights, start from zero
  python3 generations.py train --seconds 300 --every 1000
                                           continue training (time-bounded, resumable);
                                           saves generations/gen_<epoch>.pkl every
                                           --every epochs
  python3 generations.py league --seconds 3600 --every 100 [--pool generations_9std]
                                           league training: part self-play, part one side
                                           learning against FROZEN past generations
                                           (--self-frac 0.3, --pfsp 1.0). With --tag, the
                                           untagged folder of the same mode is the default pool.
  python3 generations.py crossplay --games 300 [--seconds 300] [--pool DIR] [--sort]
                                           (--pool also plays the generations of another
                                           folder, shown as o_...; their finished pairings
                                           are reused from that folder's crossplay cache)
                                           round-robin: every generation's VILLAGE side
                                           against every generation's WOLF side
                                           (wolf side = wolf_* + madman_* policies).
                                           Resumable: finished pairings are cached.
  python3 generations.py report            print the strength table from the cache

Why cross-play: self-play win rate is NOT a strength measure (village and
wolves co-evolve, so it just oscillates). A generation is stronger if its
village beats other generations' wolves, and its wolves beat other
generations' villages.

Files:
  generations/zero.pkl            all-zero weights = uniform-random baseline
  generations/legacy_*.pkl        the weights backed up by `reset`
  generations/gen_XXXXXX.pkl      generation snapshots
  generations/crossplay_cache.json  finished pairings (wins, games)
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys
import time

# --mode must be applied BEFORE config / persistence are imported.
for _flag, _env in (("--mode", "WEREWOLF_MODE"), ("--tag", "WEREWOLF_TAG")):
    for _i, _a in enumerate(sys.argv):
        if _a == _flag and _i + 1 < len(sys.argv):
            os.environ[_env] = sys.argv[_i + 1]
            del sys.argv[_i:_i + 2]
            break

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config  # noqa: E402
GEN_DIR = os.path.join(HERE, config.GEN_DIR_NAME)
CACHE = os.path.join(GEN_DIR, "crossplay_cache.json")
WOLF_SIDE = ("wolf_", "madman_")
# Weight-key prefixes of the village side (everything that is not wolf/madman;
# the critic "value_function" is deliberately in neither group).
VILLAGE_SIDE = ("village_", "medium_", "hunter_", "seer_", "villager_")
# side names understood by self_play.update_policies / decay_all_policies
WOLF_SIDES = ("wolf", "madman")
VILLAGE_SIDES = ("village", "medium", "hunter", "seer", "villager")

# Optional second folder of generation files (set by --pool). Its files are
# shown with an "o_" prefix so they never clash with this run's own names.
POOL_DIR = None
IMPORTED_KEYS = set()

import persistence  # noqa: E402


def _ensure_dir():
    os.makedirs(GEN_DIR, exist_ok=True)


# ------------------------------------------------------------ reset
def cmd_reset(args):
    _ensure_dir()
    live = os.path.join(HERE, persistence.WEIGHTS_FILE)
    if os.path.exists(live):
        persistence.load_weights(live)
        tag = f"legacy_{persistence.total_epochs:06d}.pkl"
        shutil.copy2(live, os.path.join(GEN_DIR, tag))
        print(f"backed up current weights -> generations/{tag}")
    print("This will replace trained_weights.pkl with all-zero weights.")
    if not args.yes:
        print("Re-run with --yes to confirm.")
        return
    # Fresh process state is needed: re-exec so module policies are zero.
    # (subprocess, not os.execv: execv does not quote arguments on Windows,
    # so a folder name with spaces such as "werewolf_ai (1)" broke it.)
    import subprocess
    subprocess.run([sys.executable, os.path.abspath(__file__), "_write_zero"], check=True)


def cmd__write_zero(args):
    # Executed in a fresh interpreter: nothing has been loaded, so every
    # policy still has its zero-initialised weights.
    _ensure_dir()
    persistence.total_epochs = 0
    persistence.total_games = 0
    zero = os.path.join(GEN_DIR, "zero.pkl")
    persistence.save_weights(zero)
    persistence.save_weights(os.path.join(HERE, persistence.WEIGHTS_FILE))
    shutil.copy2(zero, os.path.join(GEN_DIR, "gen_000000.pkl"))
    print("reset done: " + persistence.WEIGHTS_FILE + " is now all zeros (epoch 0).")


# ------------------------------------------------------------ train
def cmd_train(args):
    import self_play
    import train_and_report as tr

    _ensure_dir()
    persistence.load_weights(os.path.join(HERE, persistence.WEIGHTS_FILE))
    deadline = time.time() + args.seconds
    G = args.games or tr.GAMES_PER_EPOCH
    start = persistence.total_epochs
    done = 0
    last_save = time.time()
    try:
        while time.time() < deadline:
            results = self_play.run_self_play(G, verbose=False)
            self_play.update_policies(
                results, learning_rate=(args.lr or tr.LEARNING_RATE),
                use_value_function=tr.USE_VALUE_FUNCTION,
                value_learning_rate=tr.VALUE_LEARNING_RATE)
            self_play.decay_all_policies(decay_rate=tr.DECAY_RATE)
            persistence.total_epochs += 1
            persistence.total_games += G
            done += 1
            vw = sum(1 for r in results if r["winner"] == "Village") / len(results) * 100
            left = max(0, deadline - time.time())
            print(f"epoch {persistence.total_epochs} (this run: {done})  "
                  f"village win {vw:.1f}% (noisy)  time left {left:.0f}s", flush=True)
            if persistence.total_epochs % args.every == 0:
                path = os.path.join(GEN_DIR, f"gen_{persistence.total_epochs:06d}.pkl")
                persistence.save_weights(path)
            if time.time() - last_save > 300:
                persistence.save_weights(os.path.join(HERE, persistence.WEIGHTS_FILE))
                last_save = time.time()
    finally:
        persistence.save_weights(os.path.join(HERE, persistence.WEIGHTS_FILE))
    print(f"trained {done} epochs ({start} -> {persistence.total_epochs})")


# ------------------------------------------------------------ league
def _quiet(fn, *a, **k):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _resolve_dir(d):
    if not d or d.lower() == "none":
        return None
    d = d if os.path.isabs(d) else os.path.join(HERE, d)
    return d if os.path.isdir(d) else None


def _default_pool_dir():
    """With --tag, the untagged folder of the same mode (e.g. generations_9std)
    is the natural sparring-partner pool."""
    if config.TAG:
        return _resolve_dir("generations" if config.ROLE_ABSENCE else "generations_9std")
    return None


def _league_pool(pool_dir):
    """Frozen sparring partners: this run's own saved generations, the zero
    baseline and any legacy backup, plus every generation file in pool_dir."""
    files = []
    for d in (GEN_DIR, pool_dir):
        if d and os.path.isdir(d):
            files += sorted(glob.glob(os.path.join(d, "gen_*.pkl")))
            files += sorted(glob.glob(os.path.join(d, "legacy_*.pkl")))
    zero = os.path.join(GEN_DIR, "zero.pkl")
    if os.path.exists(zero):
        files.append(zero)
    seen, out = set(), []
    for f in files:
        a = os.path.abspath(f)
        # gen_000000 is the same all-zero weights as zero.pkl
        if a in seen or a.endswith("gen_000000.pkl"):
            continue
        seen.add(a)
        out.append(f)
    return out


def cmd_league(args):
    """Self-play + sparring against FROZEN past generations.

    Every epoch is one of
      self   : both sides learn (normal co-evolution)
      village: the village side learns against a frozen wolf side taken from a
               past generation (wolf side = wolf_* + madman_* policies)
      wolf   : the wolf side learns against a frozen village side
    Opponents come from the pool (own saved generations + --pool folder) and
    are drawn so that the ones the learner still loses to appear more often.
    Why: pure self-play lets each side overfit to its current opponent (the
    village can end up worse than random once the wolves learn to exploit it).
    """
    global POOL_DIR
    import self_play
    import train_and_report as tr

    _ensure_dir()
    live = os.path.join(HERE, persistence.WEIGHTS_FILE)
    persistence.load_weights(live)
    pool_dir = _resolve_dir(args.pool) if args.pool else _default_pool_dir()
    POOL_DIR = pool_dir
    pool = _league_pool(pool_dir)
    G = args.games or tr.GAMES_PER_EPOCH
    lr = args.lr or tr.LEARNING_RATE
    tmp = os.path.join(GEN_DIR, "_league_tmp.pkl")
    est = {}      # (opponent path, learner side) -> running estimate of the learner's win rate
    bad = set()   # opponents that could not be loaded
    deadline = time.time() + args.seconds
    start = persistence.total_epochs
    done = 0
    last_save = time.time()
    print(f"league: {len(pool)} frozen opponents in the pool"
          + (f" (including folder {os.path.basename(pool_dir)})" if pool_dir else "")
          + f", {int(args.self_frac * 100)}% pure self-play")
    try:
        while time.time() < deadline:
            usable = [p for p in pool if p not in bad]
            swapped = None   # weight prefixes currently overwritten by a frozen opponent
            kind, opp, freeze = "self", None, ()
            try:
                if usable and random.random() >= args.self_frac:
                    kind = random.choice(("village", "wolf"))
                    wts = [(1.0 - est.get((p, kind), 0.5)) ** args.pfsp + 0.05 for p in usable]
                    opp = random.choices(usable, weights=wts)[0]
                    frozen, freeze = (WOLF_SIDE, WOLF_SIDES) if kind == "village" \
                        else (VILLAGE_SIDE, VILLAGE_SIDES)
                    _quiet(persistence.save_weights, tmp)   # our own weights, to restore afterwards
                    swapped = frozen
                    if not _quiet(persistence.load_weights, opp, prefixes=frozen):
                        bad.add(opp)
                        print(f"  (skipping unusable opponent {os.path.basename(opp)})")
                        continue
                results = self_play.run_self_play(G, verbose=False)
                self_play.update_policies(
                    results, learning_rate=lr, freeze_sides=freeze,
                    use_value_function=tr.USE_VALUE_FUNCTION,
                    value_learning_rate=tr.VALUE_LEARNING_RATE)
                self_play.decay_all_policies(decay_rate=tr.DECAY_RATE, freeze_sides=freeze)
            finally:
                if swapped:
                    # put our own copy of the frozen side back (also on Ctrl+C)
                    _quiet(persistence.load_weights, tmp, prefixes=swapped)
            persistence.total_epochs += 1
            persistence.total_games += G
            done += 1
            vw = sum(1 for r in results if r["winner"] == "Village") / len(results)
            if kind == "self":
                label = "self-play"
            else:
                learner_wr = vw if kind == "village" else 1.0 - vw
                key = (opp, kind)
                est[key] = 0.7 * est.get(key, 0.5) + 0.3 * learner_wr
                label = f"{kind} learns vs {_short(_name(opp))} " + \
                        ("wolves" if kind == "village" else "villagers")
            left = max(0, deadline - time.time())
            print(f"epoch {persistence.total_epochs} (this run: {done})  [{label}]  "
                  f"village win {vw * 100:.1f}%  time left {left:.0f}s", flush=True)
            if persistence.total_epochs % args.every == 0:
                persistence.save_weights(
                    os.path.join(GEN_DIR, f"gen_{persistence.total_epochs:06d}.pkl"))
                pool = _league_pool(pool_dir)   # the new generation joins the pool
            if time.time() - last_save > 300:
                persistence.save_weights(live)
                last_save = time.time()
    finally:
        persistence.save_weights(live)
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"trained {done} epochs ({start} -> {persistence.total_epochs})")


# -------------------------------------------------------- crossplay
def _set_pool(args):
    global POOL_DIR
    POOL_DIR = _resolve_dir(getattr(args, "pool", None)) if getattr(args, "pool", None) \
        else (_default_pool_dir() if getattr(args, "cmd", "") == "league" else None)


def _gen_files(args):
    files = sorted(glob.glob(os.path.join(GEN_DIR, "gen_*.pkl")))
    if args.stride > 1:
        files = files[::args.stride]
    extra = []
    zero = os.path.join(GEN_DIR, "zero.pkl")
    if os.path.exists(zero):
        extra.append(zero)
    extra += sorted(glob.glob(os.path.join(GEN_DIR, "legacy_*.pkl")))
    # the zero baseline equals gen_000000; do not list it twice
    files = [f for f in files if not f.endswith("gen_000000.pkl")]
    pooled = []
    if POOL_DIR:
        pf = sorted(glob.glob(os.path.join(POOL_DIR, "gen_*.pkl")))
        if args.stride > 1:
            pf = pf[::args.stride]
        pooled = [f for f in pf if not f.endswith("gen_000000.pkl")]
    return extra + files + pooled


def _short(name):
    if name.startswith("o_"):
        return "o_" + _short(name[2:])
    if name.startswith("gen_"):
        return "g" + str(int(name[4:]))
    if name.startswith("legacy_"):
        return "legacy"
    return name


def _name(path):
    base = os.path.basename(path).replace(".pkl", "")
    if POOL_DIR and os.path.abspath(os.path.dirname(path)) == os.path.abspath(POOL_DIR):
        return "o_" + base
    return base


def _load_cache():
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            cache = json.load(f)
    IMPORTED_KEYS.clear()
    if POOL_DIR:
        pc = os.path.join(POOL_DIR, "crossplay_cache.json")
        if os.path.exists(pc):
            # reuse the finished pairings of the old folder (names get the o_ prefix)
            with open(pc) as f:
                old = json.load(f)
            for k, v in old.items():
                a, b, n = k.split("|")
                a = a if a == "zero" else "o_" + a
                b = b if b == "zero" else "o_" + b
                nk = f"{a}|{b}|{n}"
                if nk not in cache:
                    cache[nk] = v
                    IMPORTED_KEYS.add(nk)
    return cache


def _save_cache(cache):
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({k: v for k, v in cache.items() if k not in IMPORTED_KEYS}, f, indent=1)
    os.replace(tmp, CACHE)


def _play_pair(vpath, wpath, games):
    import main
    persistence.load_weights(vpath)
    persistence.load_weights(wpath, prefixes=WOLF_SIDE)
    wins = 0
    for _ in range(games):
        if main.run_game(verbose=False)["winner"] == "Village":
            wins += 1
    return wins


def cmd_crossplay(args):
    _ensure_dir()
    _set_pool(args)
    files = _gen_files(args)
    if len(files) < 2:
        print("need at least 2 weight files in generations/")
        return
    cache = _load_cache()
    deadline = time.time() + args.seconds if args.seconds else None
    todo = [(v, w) for v in files for w in files
            if f"{_name(v)}|{_name(w)}|{args.games}" not in cache]
    print(f"{len(files)} weight sets, {len(todo)} pairings left "
          f"({args.games} games each)")
    for v, w in todo:
        if deadline and time.time() > deadline:
            print("time limit reached; rerun to continue.")
            break
        key = f"{_name(v)}|{_name(w)}|{args.games}"
        wins = _play_pair(v, w, args.games)
        cache[key] = wins
        _save_cache(cache)
        print(f"  village={_name(v)} vs wolf={_name(w)}: village wins {wins}/{args.games}")
    cmd_report(args, files=files)


def cmd_report(args, files=None):
    _set_pool(args)
    files = files or _gen_files(args)
    cache = _load_cache()
    names = [_name(f) for f in files]
    g = args.games

    def rate(v, w):
        k = f"{v}|{w}|{g}"
        return None if k not in cache else cache[k] / g

    if args.matrix or len(names) <= 20:
        print("\nVillage win rate  (rows = village generation, cols = wolf generation)")
        print(f"{'':>14}" + "".join(f"{_short(n):>10}" for n in names))
        for v in names:
            row = [rate(v, w) for w in names]
            print(f"{v:>14}" + "".join(f"{'   -   ':>10}" if x is None else f"{x*100:9.1f}%" for x in row))
    else:
        print(f"\n(matrix hidden: {len(names)} generations are too wide for a terminal; "
              "add --matrix to print it anyway)")

    print("\nStrength (mean over all opponent generations; 50% = even;"
          " worst = lowest result against any single opponent generation)")
    print(f"{'generation':>16} {'village':>9} {'wolf':>9} {'overall':>9} {'worst(v)':>9} {'worst(w)':>9}")
    rows = []
    for n in names:
        vs = [rate(n, w) for w in names if rate(n, w) is not None]
        ws = [1 - rate(v, n) for v in names if rate(v, n) is not None]
        if not vs or not ws:
            continue
        vs_m, ws_m = sum(vs) / len(vs), sum(ws) / len(ws)
        rows.append((n, vs_m, ws_m, (vs_m + ws_m) / 2, min(vs), min(ws)))
    if args.sort:
        rows.sort(key=lambda r: -r[3])
    for n, vs_m, ws_m, ov, vmin, wmin in rows:
        print(f"{n:>16} {vs_m*100:8.1f}% {ws_m*100:8.1f}% {ov*100:8.1f}% {vmin*100:8.1f}% {wmin*100:8.1f}%")


def main_cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reset"); p.add_argument("--yes", action="store_true")
    sub.add_parser("_write_zero")
    p = sub.add_parser("train")
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--every", type=int, default=1000)
    p.add_argument("--games", type=int, default=0)
    p.add_argument("--lr", type=float, default=0,
                   help="learning rate (default: train_and_report.LEARNING_RATE = 0.03)")
    p = sub.add_parser("league")
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--every", type=int, default=100)
    p.add_argument("--games", type=int, default=0)
    p.add_argument("--lr", type=float, default=0)
    p.add_argument("--pool", default="",
                   help="folder of past generations to spar against "
                        "(default with --tag: the untagged folder of the same mode; 'none' = off)")
    p.add_argument("--self-frac", dest="self_frac", type=float, default=0.3,
                   help="share of epochs that are plain self-play (default 0.3)")
    p.add_argument("--pfsp", type=float, default=1.0,
                   help="how strongly to favour opponents the learner still loses to (0 = uniform)")
    for name in ("crossplay", "report"):
        p = sub.add_parser(name)
        p.add_argument("--games", type=int, default=300)
        p.add_argument("--seconds", type=float, default=0)
        p.add_argument("--stride", type=int, default=1,
                       help="use every Nth generation file")
        p.add_argument("--pool", default="",
                       help="also include the generation files of this folder (shown as o_...)")
        p.add_argument("--sort", action="store_true", help="sort the strength table by overall")
        p.add_argument("--matrix", action="store_true", help="print the full matrix even if wide")
    args = ap.parse_args()
    {"reset": cmd_reset, "_write_zero": cmd__write_zero, "train": cmd_train,
     "league": cmd_league, "crossplay": cmd_crossplay, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main_cli()
