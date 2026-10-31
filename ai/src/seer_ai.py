import policy_utils
import board_split


# ============================================================
# Seer AI (占い師)
#
# Unlike the Medium (whose ability is PASSIVE -- a report about
# whoever the village already executed, arriving one day late) the
# Seer's ability is PROACTIVE: every night, they privately choose one
# living player to investigate and learn that player's true
# alignment (wolf / not-wolf) immediately -- before anyone else has
# any public information at all about that player.
#
# CO TIMING IS NO LONGER A LEARNED CHOICE. The Seer used to be free to
# choose when (or whether) to reveal -- staying silent kept good
# information unprotected but hid a high-value kill target from the
# wolves, while revealing immediately handed them exactly that target
# on night one. At the user's explicit request, this has been changed
# to match standard human convention: the Seer is now REQUIRED to
# reveal on Day 1, exactly like the Medium (main.py's medium_day_co /
# seer_day_co both force this unconditionally). One structural side
# effect of this change: it makes a "Sy" claim on any day OTHER than
# Day 1 just as impossible for the real Seer as a late "Ps" claim
# already was for the real Medium -- see village_ai's
# _is_impossible_late_forced_claim, which now covers both roles.
#
# The old CO-timing policy (choose_co_action / co_policy / co_lookup)
# had zero call sites once the Day-1-forced-reveal change landed and
# was removed in a cleanup pass (2026); an earlier delivered zip has
# it back if this timing decision is ever reinstated as a real choice.
#
# What's left is the one decision the Seer still makes:
#   choose_investigate_target -- who to investigate tonight
#                                 (cannot investigate self).
# ============================================================

NUM_FEATURES = 10  # 2026-10-06: +[5..9], see _extract_investigate_features
# [0] bias
# [1] is_already_investigated -- this candidate is already in the
#     Seer's OWN investigation history (PRIVATE). Investigating them
#     again wastes the night's one action -- alignment never changes
#     -- so this should learn a strongly negative weight once there's
#     anything else worth investigating instead.
# [2] is_gray -- this candidate hasn't claimed anything yet. An
#     unclaimed player is often the most valuable unknown to resolve,
#     since a claimed player already gives the village OTHER ways to
#     judge them (contested/sole/provably-fake) even without a Seer
#     result.
# [3] day (normalized)
# [4] is_guard_target_tonight (2026-10-04 user request): this
#     candidate is ONE of tonight's publicly announced guard-
#     instruction targets (state["guard_instructions"].values() --
#     the SAME public dict main.py prints as "護衛指示" each day).
#     Unlike is_already_investigated/the confirmed_villagers hard
#     filter in _unexplored_pool, this is NOT a certainty -- the
#     Seer still learns their alignment either way, and the attack
#     might miss them even unguarded -- so it's only a soft, LEARNED
#     deprioritization signal: investigating someone who already has
#     a decent chance of surviving the night anyway (because a watch
#     slot is pointed at them) is usually less urgent than someone
#     with no such cover, all else equal.
investigate_policy = policy_utils.LinearPolicy(NUM_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 extension). The Seer's
# free choice of investigation target only ever happens at NIGHT,
# after Day 1's CO sequence has fully settled (the Day-1 "random
# white" assignment is a forced pre-game pick, not a call into this
# policy at all) -- plain state.get("day1_config") is safe.
investigate_split = board_split.BoardSplitManager(investigate_policy)


def _extract_investigate_features(state, candidate):

    co_history = state["co_history"]
    day = state["day"]
    own_investigations = state.get("own_investigations", [])

    already_investigated = 1.0 if any(
        inv["target"] == candidate for inv in own_investigations
    ) else 0.0
    is_gray = 1.0 if co_history[candidate] is None else 0.0

    guard_instructions = state.get("guard_instructions", {})
    is_guard_target_tonight = 1.0 if candidate in guard_instructions.values() else 0.0

    return [
        1.0,
        already_investigated,
        is_gray,
        day / 7.0,
        is_guard_target_tonight,
        # 2026-10-06 additions: who is worth the night's one look.
        # [5] a rival Sy claimant, [6] Ps claimant, [7] Hn claimant
        # (a claimed role already has public judgment signals, but
        # which one to check first is a real human decision), [8] how
        # likely the village is to execute them anyway (public
        # distribution -- investigating someone about to be hanged is
        # a wasted look), [9] someone a Sy claimant has ALREADY
        # reported on (a public white/black exists).
        1.0 if (co_history[candidate] == "Sy" and candidate != state.get("self_index")) else 0.0,
        1.0 if co_history[candidate] == "Ps" else 0.0,
        1.0 if co_history[candidate] == "Hn" else 0.0,
        min(state.get("village_vote_probs", {}).get(candidate, 0.0) * max(1, len(state.get("village_vote_probs", {}))), 4.0) / 2.0,
        1.0 if any(r["player"] == candidate for r in state.get("seer_reports", [])) else 0.0,
    ]


def _unexplored_pool(state, candidates):
    """
    Filter `candidates` down to ones this Seer has NOT already
    investigated (including the mandatory pre-game "random white"
    entry -- see main.py's _assign_seer_random_white -- which lands
    in own_investigations exactly like a real night's result).

    Re-investigating an already-known target is not just usually
    bad, it is STRICTLY dominated whenever an unexplored target
    exists: a player's alignment never changes, so re-investigating
    one can NEVER produce new information, while investigating an
    unexplored player always CAN. This is a logical certainty with
    no dependency on who is a wolf/Madman/etc (unlike the hard-coded
    "only wolves fake claims" assumption that caused the
    _wolf_counting_deduction and medium_report bugs elsewhere), so
    it is safe to enforce as a hard pre-filter on the candidate pool
    -- the same apply_protection_floor-style pattern village_ai uses
    for execution -- rather than trusting the learned policy to
    discover it purely from a noisy, heavily-diluted full-Monte-Carlo
    REINFORCE signal.
    #
    # This matters in practice, not just in theory: after 59,000+
    # training epochs, the already_investigated weight had only
    # crept to about -0.08 -- the "wasted night" signal is real but
    # extremely weak per-decision (unlike, say, an execution that can
    # immediately end the game), so relying on gradient descent alone
    # to learn a habit that should really be absolute took far too
    # long and left it easy to override by chance.
    #
    # If EVERY living candidate has already been investigated (only
    # possible in a very late, small endgame), there is nothing left
    # to gain either way, so the full candidate list is returned
    # unfiltered and the learned policy (is_gray / day, etc.) picks
    # among them as before.
    """
    own_investigations = state.get("own_investigations", [])
    investigated_targets = {inv["target"] for inv in own_investigations}
    unexplored = [c for c in candidates if c not in investigated_targets]
    pool = unexplored if unexplored else candidates

    # 2026-10-04 user request: also hard-filter out anyone already
    # PUBLICLY confirmed non-wolf (state["confirmed_villagers"] --
    # a night-attack victim, a guard-verified survivor, or the
    # wolf-counting pigeonhole deduction). Investigating one of these
    # is exactly as wasted as re-investigating an old personal target
    # above: their alignment is already public with 100% certainty,
    # so the Seer's own result could only ever repeat "not a wolf" --
    # no new information is possible, by the same strict-domination
    # argument as the already-investigated filter, not a judgment
    # call. (This was previously impossible to apply safely for
    # SEER-reported "white" verdicts specifically -- see
    # build_public_state's "Seer cascade" removal note -- but
    # confirmed_villagers itself only ever contains hard, non-Seer-
    # report-derived proof, so filtering on it here was always safe;
    # it just wasn't being read at all.)
    confirmed_villagers = state.get("confirmed_villagers", [])
    cleared = [c for c in pool if c not in confirmed_villagers]
    return cleared if cleared else pool


def choose_investigate_target(state):
    """
    Decide who to investigate tonight.

    Parameters
    ----------
    state : dict
        {
            "board": list[str],
            "co_history": list,
            "day": int,
            "self_index": int,           # the seer's own index (private)
            "candidates": list[int],     # living players EXCLUDING self
            "own_investigations": list,  # PRIVATE: this seer's own past
                # {"day", "target", "is_wolf"} results, so far.
        }

    Returns
    -------
    int
        Index of the player to investigate.
    """

    candidates = state["candidates"]
    pool = _unexplored_pool(state, candidates)
    feature_list = [_extract_investigate_features(state, c) for c in pool]

    active_policy = investigate_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, pool)

    return chosen


def update(kind, state, action, reward, learning_rate=0.01):

    if kind == "investigate":

        candidates = state["candidates"]
        # Must match the SAME pool choose_investigate_target actually
        # sampled from -- see village_ai.update's "execution" branch
        # (and its long docstring) for why training against the
        # unfiltered candidate list instead of the real, filtered pool
        # corrupts the REINFORCE baseline and was a serious bug there.
        pool = _unexplored_pool(state, candidates)
        feature_list = [_extract_investigate_features(state, c) for c in pool]

        active_policy = investigate_split.active(state.get("day1_config"))
        active_policy.update(feature_list, pool, action, reward, learning_rate)
        investigate_split.record_sample(state.get("day1_config"))
