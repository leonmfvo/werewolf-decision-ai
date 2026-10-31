import math

import policy_utils


# ============================================================
# Value function (Critic)
#
# Predicts "given this game state, what's the expected outcome
# from the VILLAGE's perspective?" (a number in roughly [-1, +1],
# where +1 means "village should win from here" and -1 means
# "werewolves should win from here").
#
# WHY THIS EXISTS: self_play.py's REINFORCE update previously used
# a single running-average baseline per (side, decision type) --
# e.g. "village executions average -0.5 reward". That's a CONSTANT
# for every state, so a genuinely good decision made in a state
# that was already heavily stacked against the village (say, 2
# wolves alive vs 1 remaining villager) gets blamed almost as hard
# as a bad one, because the whole GAME was likely lost regardless
# of that specific choice. A state-dependent value function fixes
# this: the advantage becomes (actual outcome - predicted outcome
# FOR THIS SPECIFIC STATE), so a good decision in a losing position
# can still get credit for being less bad than expected, and a bad
# decision in a winning position gets properly blamed even though
# the team went on to win anyway.
#
# Since Werewolf/Village win chances are exact opposites (it's a
# two-team zero-sum game), ONE shared value function covers both
# sides: wolf-side decisions just use -predict(state) instead of
# +predict(state).
#
# Uses only the UNIVERSAL fields present in every state dict (the
# ones build_public_state always fills in), not side-specific
# extras like "candidates" or "targets" -- so the exact same
# function can score a state regardless of which side's decision
# it came from.
# ============================================================

NUM_VALUE_FEATURES = 14
# [0]  bias
# [1]  alive_count (normalized) -- "N-nin-nichi" population size
# [2]  is_medium_alive_and_revealed
# [3]  is_hunter_alive_and_revealed
# [4]  medium_claimant_count (normalized) -- contested Ps claims
# [5]  hunter_claimant_count (normalized) -- contested Hn claims
# [6]  guard_success_count (normalized)
# [7]  confirmed_villagers_count (normalized) -- via revealed guard saves
# [8]  confirmed_identified_wolves (normalized) -- via report or verification
# [9]  total_wolves (normalized) -- how many wolves this composition has
# [10] day (normalized)
# [11] has_pending_verification -- an unresolved guard-instruction contest exists
# [12] is_seer_alive_and_revealed
# [13] seer_claimant_count (normalized) -- contested Sy claims. A live,
#      revealed real Seer is a strong ongoing advantage for the
#      village (a repeatable, proactive information source, unlike
#      the Medium's one-shot-per-execution report), so this is worth
#      the critic seeing directly rather than folding it into
#      confirmed_villagers_count, which only reflects PAST results.

value_weights = [0.0] * NUM_VALUE_FEATURES


def _extract_value_features(state):

    board = state.get("board", [])
    co_history = state.get("co_history", [])
    alive_count = state.get("alive_count", sum(1 for s in board if s != "--"))
    day = state.get("day", 1)

    medium_claimant_count = sum(1 for co in co_history if co == "Ps")
    hunter_claimant_count = sum(1 for co in co_history if co == "Hn")
    seer_claimant_count = sum(1 for co in co_history if co == "Sy")

    medium_alive_revealed = any(
        co == "Ps" and i < len(board) and board[i] != "--"
        for i, co in enumerate(co_history)
    )
    hunter_alive_revealed = any(
        co == "Hn" and i < len(board) and board[i] != "--"
        for i, co in enumerate(co_history)
    )
    seer_alive_revealed = any(
        co == "Sy" and i < len(board) and board[i] != "--"
        for i, co in enumerate(co_history)
    )

    guard_count = state.get("guard_success_count", 0)
    confirmed_villagers_count = len(state.get("confirmed_villagers", []))
    confirmed_identified_wolves = state.get("confirmed_dead_wolves", 0)
    total_wolves = state.get("total_wolves", 2)
    has_pending_verification = 1.0 if state.get("guard_instructions") else 0.0

    return [
        1.0,
        alive_count / 7.0,
        1.0 if medium_alive_revealed else 0.0,
        1.0 if hunter_alive_revealed else 0.0,
        medium_claimant_count / 3.0,
        hunter_claimant_count / 3.0,
        guard_count / 2.0,
        confirmed_villagers_count / 3.0,
        confirmed_identified_wolves / max(total_wolves, 1),
        total_wolves / 3.0,
        day / 7.0,
        has_pending_verification,
        1.0 if seer_alive_revealed else 0.0,
        seer_claimant_count / 3.0,
    ]


def predict(state, side):
    """
    Predicted expected outcome for `side`, in roughly [-1, +1].

    Parameters
    ----------
    state : dict
        Any state dict built via main.py's build_public_state
        (extra side-specific keys are simply ignored).
    side : str
        "wolf", or any Village-aligned side ("village", "medium",
        "hunter", "villager") -- determines which perspective the
        returned value is from.
    """
    features = _extract_value_features(state)
    raw = sum(w * f for w, f in zip(value_weights, features))
    village_perspective = math.tanh(raw)  # squash to (-1, 1)
    return -village_perspective if side in ("wolf", "madman") else village_perspective


def update(state, village_outcome, learning_rate=0.01):
    """
    One gradient step toward correctly predicting `village_outcome`
    (+1.0 if Village won the game this state came from, -1.0 if
    Werewolf won) from this state.

    Uses squared-error gradient descent through the tanh squashing:
    d/dw [ (target - tanh(w.x))^2 ] = -2*(target - tanh(w.x))*(1-tanh(w.x)^2)*x
    """
    features = _extract_value_features(state)
    raw = sum(w * f for w, f in zip(value_weights, features))
    predicted = math.tanh(raw)

    error = village_outcome - predicted
    dtanh = 1.0 - predicted * predicted  # tanh'(raw)

    for i in range(len(value_weights)):
        value_weights[i] += learning_rate * error * dtanh * features[i]
