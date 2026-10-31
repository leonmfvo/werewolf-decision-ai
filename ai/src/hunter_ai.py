import policy_utils
import board_split


# ============================================================
# Hunter AI
#
# Two independent decisions:
#   1. choose_co_action  -- reveal as Hunter (once, Day 1) or stay
#                            silent. Unlike the Medium, there is
#                            currently no way for a wolf to fake
#                            this claim, so an "Hn" claim is always
#                            genuine -- revealing trades "safer from
#                            execution" for "a clearer night target".
#   2. choose_guard_target -- who to protect each night (cannot
#                              guard self).
#
# Both are linear-softmax policies. Weights start at zero.
# ============================================================

CO_OPTIONS = ["reveal", "silent"]

# HISTORY: this used to be a lookup table over (guard_success_count,
# gray_count). Weight decay now handles the "stale belief" problem
# that motivated tables in the first place, so this is back to a
# plain linear model -- one-hot option identity, with guard_count /
# gray_count / under_threat as continuous features interacted per
# option (so each option gets its own independently-learnable
# sensitivity to each one).
NUM_CO_FEATURES = 2 + 2 * 4
# [reveal, silent,
#  guard_count*reveal, guard_count*silent,
#  gray_count*reveal, gray_count*silent,
#  under_threat*reveal, under_threat*silent,
#  personally_nominated*reveal, personally_nominated*silent]

co_policy = policy_utils.LinearPolicy(NUM_CO_FEATURES)

# NOT split by board config (2026-10 user decision -- see wolf_ai.
# co_policy's matching comment): this is a reveal/stay-silent timing
# decision, one of the things that PRODUCES the Day-1 board
# configuration, so splitting it BY that configuration would be
# backwards. Stays a single shared policy.


def _co_feature_list(state):

    guard_count = state.get("guard_success_count", 0) / 2.0
    gray_count = state.get("gray_count", 0) / 5.0
    under_threat = 1.0 if state.get("under_threat", False) else 0.0
    personally_nominated = 1.0 if state.get("personally_nominated", False) else 0.0

    features = []
    for j in range(2):
        f = [0.0] * NUM_CO_FEATURES
        f[j] = 1.0
        f[2 + j] = guard_count
        f[4 + j] = gray_count
        f[6 + j] = under_threat
        f[8 + j] = personally_nominated
        features.append(f)

    return features


def choose_co_action(state):
    """
    Decide whether to claim Hunter THIS day (only asked while
    still silent and alive -- see main.py's hunter_day_co).

    Parameters
    ----------
    state : dict
        {
            "board": ..., "co_history": ..., "day": ...,
            "guard_success_count": int,  # PUBLIC: how many guards
                # have succeeded so far this game.
            "gray_count": int,           # PUBLIC: how many living
                # players are still unclaimed right now.
            "under_threat": bool,        # PRIVATE: is the village
                # about to execute me RIGHT NOW, before I've revealed?
        }

    Returns
    -------
    str
        One of "reveal", "silent"
    """

    feature_list = _co_feature_list(state)
    chosen, _probs = co_policy.choose(feature_list, CO_OPTIONS)

    return chosen


NUM_FEATURES = 7  # 2026-10-06: +[5] was_last_guard, [6] is_confirmed_villager
# [0] bias
# [1] is_psychic_claimant
# [2] day (normalized)
# [3] is_seer_claimant       -- this candidate has claimed Sy (mirrors
#     is_psychic_claimant -- protecting a role that keeps producing
#     new public information every day it survives is valuable
#     regardless of whether it turns out to be genuine, the same way
#     protecting a Medium claimant already was).
# [4] is_seer_cleared_white  -- some CURRENTLY REVEALED Sy claimant
#     has publicly reported this candidate as NOT a wolf
#     (main.py's seer_reports). This is a PUBLIC, soft signal -- the
#     Hunter has no private way to tell a real Sy claimant's report
#     from a fake one any more than the village does (see
#     is_accused_by_sole_seer/is_accused_by_contested_seer in
#     village_ai.py for the same distinction made explicit there) --
#     so this deliberately does NOT try to weight sole vs. contested
#     differently; it's one soft feature and training decides how
#     much a "cleared white" declaration is worth protecting.

guard_policy = policy_utils.LinearPolicy(NUM_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 extension). The guard
# choice is always made at NIGHT, after Day 1's CO sequence has fully
# settled -- plain state.get("day1_config") is safe (no
# safe_day1_config() guard needed, unlike co_split above).
guard_split = board_split.BoardSplitManager(guard_policy)


def _extract_features(state, candidate):

    co_history = state["co_history"]
    day = state["day"]

    is_ps_claimant = 1.0 if co_history[candidate] == "Ps" else 0.0
    is_sy_claimant = 1.0 if co_history[candidate] == "Sy" else 0.0

    seer_reports = state.get("seer_reports", [])
    is_seer_cleared_white = 1.0 if any(
        r["player"] == candidate and not r["is_wolf"]
        and co_history[r["claimant"]] == "Sy"
        for r in seer_reports
    ) else 0.0

    return [
        1.0,
        is_ps_claimant,
        day / 7.0,
        is_sy_claimant,
        is_seer_cleared_white,
        # 2026-10-06: [5] the player guarded LAST night (consecutive
        # guard is a real human-game consideration), [6] publicly
        # proven non-wolf (guarding them is rarely the best use).
        1.0 if candidate == state.get("last_guard_target") else 0.0,
        1.0 if candidate in state.get("confirmed_villagers", []) else 0.0,
    ]


def choose_guard_target(state):
    """
    Decide who to guard tonight.

    Parameters
    ----------
    state : dict
        {
            "board": list[str],
            "co_history": list,
            "day": int,
            "self_index": int,        # the hunter's own index (private)
            "candidates": list[int],  # living players EXCLUDING self
        }

    Returns
    -------
    int
        Index of the player to guard.
    """

    candidates = state["candidates"]
    feature_list = [_extract_features(state, c) for c in candidates]

    active_policy = guard_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, candidates)

    return chosen


def update(kind, state, action, reward, learning_rate=0.01):

    if kind == "co_action":

        feature_list = _co_feature_list(state)
        co_policy.update(feature_list, CO_OPTIONS, action, reward, learning_rate)

    elif kind == "guard":

        candidates = state["candidates"]
        feature_list = [_extract_features(state, c) for c in candidates]

        active_policy = guard_split.active(state.get("day1_config"))
        active_policy.update(feature_list, candidates, action, reward, learning_rate)
        guard_split.record_sample(state.get("day1_config"))
