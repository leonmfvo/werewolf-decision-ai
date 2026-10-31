"""
Save/load all AI modules' learned weights, so training can be
stopped and resumed instead of starting from zero every run.
"""

import config
import os
import pickle

import wolf_ai
import village_ai
import medium_ai
import hunter_ai
import seer_ai
import villager_ai
import madman_ai
import value_function
import board_split


WEIGHTS_FILE = config.WEIGHTS_FILE  # mode-specific, see config.MODE

total_epochs = 0  # cumulative epochs trained across all runs so far
total_games = 0   # cumulative self-play games across all runs so far


# Policies whose feature layout only ever grows at the END (see the
# padding note in load_weights).
_APPEND_ONLY_KEYS = {
    "village_exec", "medium_nomination_target", "madman_vote",
    "hunter_guard", "seer_investigate", "wolf_vote", "wolf_attack",
    "wolf_accuse_target", "madman_accuse_target",
    "wolf_fake_report", "madman_fake_report", "wolf_fake_declaration",
}


def save_weights(path=WEIGHTS_FILE):
    weights = {
        "wolf_co": list(wolf_ai.co_policy.weights),
        "wolf_attack": list(wolf_ai.attack_policy.weights),
        "wolf_vote": list(wolf_ai.vote_policy.weights),
        "wolf_fake_declaration": list(wolf_ai.fake_declaration_policy.weights),
        "wolf_fake_report": list(wolf_ai.fake_report_policy.weights),
        "wolf_fake_seer_report": list(wolf_ai.fake_seer_report_policy.weights),
        "wolf_accuse_target": list(wolf_ai.accuse_target_policy.weights),
        "wolf_endgame_declare": list(wolf_ai.endgame_declare_policy.weights),
        "wolf_day1_joint_co": list(wolf_ai.day1_joint_co_policy.weights),
        "village_exec": list(village_ai.execution_policy.weights),
        "village_guard_instruction": list(village_ai.guard_instruction_policy.weights),
        "village_guard_instruction_1": list(village_ai.guard_instruction_policy_1.weights),
        "village_guard_instruction_3": list(village_ai.guard_instruction_policy_3.weights),
        "medium_nomination": list(medium_ai.nomination_policy.weights),
        "medium_nomination_target": list(medium_ai.nomination_target_policy.weights),
        "hunter_co": list(hunter_ai.co_policy.weights),
        "hunter_guard": list(hunter_ai.guard_policy.weights),
        "seer_investigate": list(seer_ai.investigate_policy.weights),
        "villager_endgame_bluff": list(villager_ai.endgame_bluff_policy.weights),
        "madman_vote": list(madman_ai.vote_policy.weights),
        "madman_co": list(madman_ai.co_policy.weights),
        "madman_fake_report": list(madman_ai.fake_report_policy.weights),
        "madman_fake_seer_report": list(madman_ai.fake_seer_report_policy.weights),
        "madman_accuse_target": list(madman_ai.accuse_target_policy.weights),
        "madman_endgame_reveal": list(madman_ai.endgame_reveal_policy.weights),
        "value_function": list(value_function.value_weights),
        "total_epochs": total_epochs,
        "total_games": total_games,
        # Per-Day-1-board-config weight sets (see board_split.py).
        # "village_exec"/"wolf_attack" above already save the SHARED
        # base policy each of these wraps; this saves the split state
        # (whether enable_board_split() has been called) plus each
        # bucket's own weights on top of that.
        "village_exec_split": village_ai.execution_split.snapshot(),
        "wolf_attack_split": wolf_ai.attack_split.snapshot(),
        # 2026-10 "split everything" extension -- every OTHER policy
        # now also wrapped in its own board_split.BoardSplitManager
        # (see board_split.py's module docstring and each AI module's
        # own comment by its *_split definition). Each one needs the
        # exact same snapshot()/load_snapshot() treatment as village_
        # exec_split/wolf_attack_split above, or its split state
        # (and every bucket's learned weights) would silently be lost
        # on every save/load cycle even though it's being actively
        # trained during the run.
        "wolf_vote_split": wolf_ai.vote_split.snapshot(),
        "wolf_fake_declaration_split": wolf_ai.fake_declaration_split.snapshot(),
        "wolf_fake_report_split": wolf_ai.fake_report_split.snapshot(),
        "wolf_fake_seer_report_split": wolf_ai.fake_seer_report_split.snapshot(),
        "wolf_accuse_target_split": wolf_ai.accuse_target_split.snapshot(),
        "wolf_endgame_declare_split": wolf_ai.endgame_declare_split.snapshot(),
        "village_guard_instruction_split": village_ai.guard_instruction_split.snapshot(),
        "village_guard_instruction_split_1": village_ai.guard_instruction_split_1.snapshot(),
        "village_guard_instruction_split_3": village_ai.guard_instruction_split_3.snapshot(),
        "medium_nomination_split": medium_ai.nomination_split.snapshot(),
        "medium_nomination_target_split": medium_ai.nomination_target_split.snapshot(),
        "hunter_guard_split": hunter_ai.guard_split.snapshot(),
        "seer_investigate_split": seer_ai.investigate_split.snapshot(),
        "villager_endgame_bluff_split": villager_ai.endgame_bluff_split.snapshot(),
        "madman_vote_split": madman_ai.vote_split.snapshot(),
        "madman_fake_report_split": madman_ai.fake_report_split.snapshot(),
        "madman_fake_seer_report_split": madman_ai.fake_seer_report_split.snapshot(),
        "madman_accuse_target_split": madman_ai.accuse_target_split.snapshot(),
        "madman_endgame_reveal_split": madman_ai.endgame_reveal_split.snapshot(),
        # board_split.FULL_SPLIT_MODE is a single module-level flag
        # shared by BOTH managers (not part of either .snapshot()
        # above), so it needs its own top-level entry here -- without
        # this, enable_full_board_split() would silently revert to
        # the grouped default (only COMMON_DAY1_CONFIGS get their own
        # bucket) every time the process restarts, even though
        # split_enabled/the buckets themselves persisted correctly.
        "board_split_full_mode": board_split.FULL_SPLIT_MODE,
        "mode": config.MODE,
    }
    # 2026-10-06: atomic save -- write to a temp file in the same
    # directory, then rename over the live file, so an interrupt in the
    # middle of the write can never leave a truncated/corrupt weights
    # file behind.
    import os, tempfile
    _dir = os.path.dirname(os.path.abspath(path))
    _fd, _tmp = tempfile.mkstemp(prefix=".weights_", suffix=".tmp", dir=_dir)
    try:
        with os.fdopen(_fd, "wb") as f:
            pickle.dump(weights, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(_tmp, path)
    except BaseException:
        try:
            os.unlink(_tmp)
        except OSError:
            pass
        raise
    print(f"[persistence] saved weights to {path} "
          f"(total_epochs={total_epochs}, total_games={total_games})")


def load_weights(path=WEIGHTS_FILE, prefixes=None):
    """`prefixes`: if given (e.g. ("wolf_", "madman_")), ONLY the policies
    / split managers whose saved key starts with one of them are loaded,
    leaving every other policy as it currently is. Used by generations.py
    to put one generation's village side against another's wolf side."""
    def _want(key):
        return prefixes is None or any(key.startswith(pf) for pf in prefixes)

    if not os.path.exists(path):
        print(f"[persistence] no existing weights at {path} -- starting fresh")
        return False

    with open(path, "rb") as f:
        weights = pickle.load(f)

    saved_mode = weights.get("mode", "absence")
    if saved_mode != config.MODE:
        print(f"[persistence] REFUSING to load {path}: it was trained in mode "
              f"'{saved_mode}' but this run is mode '{config.MODE}'. "
              "(Use the matching --mode / WEREWOLF_MODE.)")
        return False

    # If a policy's feature layout has changed since these weights
    # were saved (e.g. a new feature added), that ONE saved list will
    # have the wrong length. Loading it anyway would silently corrupt
    # the policy (or crash later with an out-of-range index), so each
    # policy is checked and loaded INDEPENDENTLY -- a shape mismatch
    # in one (say, village_exec after adding a feature) resets only
    # that one policy to its fresh zero-init state, rather than
    # throwing away every OTHER policy's perfectly-compatible,
    # already-trained weights too just because they happened to be
    # saved in the same file.
    targets = {
        "wolf_co": wolf_ai.co_policy,
        "wolf_attack": wolf_ai.attack_policy,
        "wolf_vote": wolf_ai.vote_policy,
        "wolf_fake_declaration": wolf_ai.fake_declaration_policy,
        "wolf_fake_report": wolf_ai.fake_report_policy,
        "wolf_fake_seer_report": wolf_ai.fake_seer_report_policy,
        "wolf_accuse_target": wolf_ai.accuse_target_policy,
        "wolf_endgame_declare": wolf_ai.endgame_declare_policy,
        "wolf_day1_joint_co": wolf_ai.day1_joint_co_policy,
        "village_exec": village_ai.execution_policy,
        "village_guard_instruction": village_ai.guard_instruction_policy,
        "village_guard_instruction_1": village_ai.guard_instruction_policy_1,
        "village_guard_instruction_3": village_ai.guard_instruction_policy_3,
        "medium_nomination": medium_ai.nomination_policy,
        "medium_nomination_target": medium_ai.nomination_target_policy,
        "hunter_co": hunter_ai.co_policy,
        "hunter_guard": hunter_ai.guard_policy,
        "seer_investigate": seer_ai.investigate_policy,
        "villager_endgame_bluff": villager_ai.endgame_bluff_policy,
        "madman_vote": madman_ai.vote_policy,
        "madman_co": madman_ai.co_policy,
        "madman_fake_report": madman_ai.fake_report_policy,
        "madman_fake_seer_report": madman_ai.fake_seer_report_policy,
        "madman_accuse_target": madman_ai.accuse_target_policy,
        "madman_endgame_reveal": madman_ai.endgame_reveal_policy,
    }

    any_loaded = False
    for key, policy in targets.items():
        if not _want(key):
            continue
        expected_len = len(policy.weights)
        saved = weights.get(key, [])
        # 2026-10-06: features added at the END of a layout (village_ai's
        # extra tail, hunter/seer/wolf appended features) keep every old
        # index valid, so a shorter saved list is zero-padded instead of
        # throwing the trained weights away.
        if 0 < len(saved) < expected_len and key in _APPEND_ONLY_KEYS:
            print(f"[persistence] '{key}': padding {len(saved)} -> {expected_len} "
                  "weights with zeros (features appended at the end).")
            saved = list(saved) + [0.0] * (expected_len - len(saved))
        if len(saved) != expected_len:
            print(f"[persistence] saved '{key}' has {len(saved)} entries but "
                  f"the current code expects {expected_len} -- the feature "
                  "layout changed since this file was saved. Starting that "
                  "one policy fresh; every other compatible policy still loads.")
            continue
        policy.weights = saved
        any_loaded = True

    if prefixes is not None:
        pass
    elif len(weights.get("value_function", [])) == len(value_function.value_weights):
        value_function.value_weights = weights["value_function"]
        any_loaded = True
    else:
        print(f"[persistence] saved 'value_function' has "
              f"{len(weights.get('value_function', []))} entries but the "
              f"current code expects {len(value_function.value_weights)} -- "
              "starting that one fresh.")

    # Per-Day-1-board-config weight sets (see board_split.py). Must
    # run AFTER the targets loop above, which is what just restored
    # village_ai.execution_policy / wolf_ai.attack_policy -- the exact
    # SAME objects execution_split.base / attack_split.base point to
    # -- so by this point .base already holds the freshly-loaded
    # shared weights, and load_snapshot only has to restore the split
    # flag and each bucket's own weights on top of that.
    if _want("village_exec_split"):
        village_ai.execution_split.load_snapshot(weights.get("village_exec_split"), pad_ok=True)
    if _want("wolf_attack_split"):
        wolf_ai.attack_split.load_snapshot(weights.get("wolf_attack_split"), pad_ok=True)
    # 2026-10 "split everything" extension -- mirrors the two lines
    # above for every newly-wrapped policy (see save_weights's matching
    # comment). Each of these ALSO runs after the targets loop, for the
    # same reason: .base on every one of these managers points at the
    # SAME object as the corresponding entry in `targets` above, which
    # has already been freshly loaded by the time these run.
    if _want("wolf_vote_split"):
        wolf_ai.vote_split.load_snapshot(weights.get("wolf_vote_split"), pad_ok=True)
    if _want("wolf_fake_declaration_split"):
        wolf_ai.fake_declaration_split.load_snapshot(weights.get("wolf_fake_declaration_split"), pad_ok=True)
    if _want("wolf_fake_report_split"):
        wolf_ai.fake_report_split.load_snapshot(weights.get("wolf_fake_report_split"), pad_ok=True)
    if _want("wolf_fake_seer_report_split"):
        wolf_ai.fake_seer_report_split.load_snapshot(weights.get("wolf_fake_seer_report_split"))
    if _want("wolf_accuse_target_split"):
        wolf_ai.accuse_target_split.load_snapshot(weights.get("wolf_accuse_target_split"), pad_ok=True)
    if _want("wolf_endgame_declare_split"):
        wolf_ai.endgame_declare_split.load_snapshot(weights.get("wolf_endgame_declare_split"))
    if _want("village_guard_instruction_split"):
        village_ai.guard_instruction_split.load_snapshot(weights.get("village_guard_instruction_split"))
    if _want("village_guard_instruction_split_1"):
        village_ai.guard_instruction_split_1.load_snapshot(weights.get("village_guard_instruction_split_1"))
    if _want("village_guard_instruction_split_3"):
        village_ai.guard_instruction_split_3.load_snapshot(weights.get("village_guard_instruction_split_3"))
    if _want("medium_nomination_split"):
        medium_ai.nomination_split.load_snapshot(weights.get("medium_nomination_split"))
    if _want("medium_nomination_target_split"):
        medium_ai.nomination_target_split.load_snapshot(weights.get("medium_nomination_target_split"), pad_ok=True)
    if _want("hunter_guard_split"):
        hunter_ai.guard_split.load_snapshot(weights.get("hunter_guard_split"), pad_ok=True)
    if _want("seer_investigate_split"):
        seer_ai.investigate_split.load_snapshot(weights.get("seer_investigate_split"), pad_ok=True)
    if _want("villager_endgame_bluff_split"):
        villager_ai.endgame_bluff_split.load_snapshot(weights.get("villager_endgame_bluff_split"))
    if _want("madman_vote_split"):
        madman_ai.vote_split.load_snapshot(weights.get("madman_vote_split"), pad_ok=True)
    if _want("madman_fake_report_split"):
        madman_ai.fake_report_split.load_snapshot(weights.get("madman_fake_report_split"), pad_ok=True)
    if _want("madman_fake_seer_report_split"):
        madman_ai.fake_seer_report_split.load_snapshot(weights.get("madman_fake_seer_report_split"))
    if _want("madman_accuse_target_split"):
        madman_ai.accuse_target_split.load_snapshot(weights.get("madman_accuse_target_split"), pad_ok=True)
    if _want("madman_endgame_reveal_split"):
        madman_ai.endgame_reveal_split.load_snapshot(weights.get("madman_endgame_reveal_split"))
    if prefixes is not None:
        print(f"[persistence] loaded only {prefixes} policies from {path}")
        return any_loaded
    board_split.FULL_SPLIT_MODE = bool(weights.get("board_split_full_mode"))

    global total_epochs, total_games
    total_epochs = weights.get("total_epochs", 0)
    total_games = weights.get("total_games", 0)

    print(f"[persistence] loaded weights from {path} "
          f"(total_epochs so far={total_epochs}, total_games so far={total_games})")
    return any_loaded
