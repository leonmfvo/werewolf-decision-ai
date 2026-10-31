import policy_utils
import village_ai
import board_split


# ============================================================
# Villager AI
#
# NOTE: this module used to also own a full-game "mura-katari"
# (village bluffing) mechanic -- a plain Villager falsely claiming
# Medium or Hunter, or staying silent (CO_OPTIONS/co_policy/
# choose_co_action). It was called from main.py's villager_day_co,
# which day_phase stopped invoking after 500+ epochs showed it never
# settling into a useful pattern -- so the whole mechanic had zero
# live call sites. Removed in a cleanup pass (2026); an earlier
# delivered zip has it back if this is ever worth revisiting. The
# narrower, currently-ENABLED relative of this same idea is the
# ENDGAME BLUFF below, which only fires once 3-4 players remain.
# ============================================================


# ============================================================
# ENDGAME BLUFF ("村騙り") -- STEP 5 of the Madman addition (see
# config.py's comment on "Kg").
#
# When only 3 or 4 players remain, a genuine Villager can ALSO falsely
# declare itself hostile to the village -- writing its own name into
# the exact same main.py list (self_declared_non_village) that the
# Madman's genuine endgame reveal uses (madman_ai.choose_endgame_
# reveal_action). This is the user's own explicit design requirement:
# without it, a "self-declared non-village" entry at 3-4 remaining
# would be an AUTOMATIC, unambiguous tell that this is the real
# Madman (nobody else would ever have a reason to say such a thing) --
# which would make the reveal mechanic backfire by handing the
# village a free, certain execution target. Letting a genuine
# Villager ALSO produce this exact same public fact, for a genuine
# village-aligned reason (see below), restores the ambiguity a real
# reveal needs to be worth anything.
#
# WHY a genuine Villager would ever WANT to do this: with the same
# public information (and the same wolf-population estimate) a real
# Villager can reason "the numbers already look bad enough here that
# if the Madman DOES reveal, the wolves win outright" -- muddying the
# water with a decoy declaration costs this Villager little (their
# own vote/suspicion profile barely changes) but forces anyone who
# sees a self_declared_non_village entry to second-guess whether
# it's the real thing. Whether this is actually worth doing, and
# under what conditions, is exactly what training has to discover --
# reward alone (a village-aligned win/loss signal) decides it, no
# hand-coded logic about when bluffing "should" happen.
#
# Shares village_ai.endgame_context_features with madman_ai's mirror-
# image decision (see that function's docstring) -- a genuine
# Villager has exactly the same public vantage point the Madman does,
# no private information either one lacks.
# ============================================================

# Four options, not two: a genuine Villager can now falsely claim any
# one of the three team identities -- "狼CO" (claim_wolf, the ORIGINAL
# "bluff_declare" option this replaced), "狂人CO" (claim_madman,
# poaching the real Madman's cover directly), or "村人CO"
# (claim_village -- an explicit, public "trust me" that costs nothing
# to say honestly but is also exactly what a hiding wolf/Madman could
# say). See main.py's endgame_claims / wolf_ai.ENDGAME_DECLARE_OPTIONS's
# comment for how a contested "狂人CO" claim becomes a hard deduction.
ENDGAME_BLUFF_OPTIONS = ["stay_silent", "claim_village", "claim_wolf", "claim_madman"]

NUM_ENDGAME_BLUFF_OPTIONS = len(ENDGAME_BLUFF_OPTIONS)
NUM_ENDGAME_BLUFF_FEATURES = (
    NUM_ENDGAME_BLUFF_OPTIONS
    + NUM_ENDGAME_BLUFF_OPTIONS * village_ai.NUM_ENDGAME_CONTEXT_FEATURES
)

endgame_bluff_policy = policy_utils.LinearPolicy(NUM_ENDGAME_BLUFF_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 "split everything"
# extension -- see board_split.py's module docstring). Only ever
# asked once alive_count is 3-4, long after Day 1 -- plain
# state.get("day1_config") is safe.
endgame_bluff_split = board_split.BoardSplitManager(endgame_bluff_policy)


def _endgame_bluff_feature_list(state):
    context = village_ai.endgame_context_features(state)
    features = []
    for j in range(NUM_ENDGAME_BLUFF_OPTIONS):
        f = [0.0] * NUM_ENDGAME_BLUFF_FEATURES
        f[j] = 1.0
        for k, value in enumerate(context):
            f[NUM_ENDGAME_BLUFF_OPTIONS + k * NUM_ENDGAME_BLUFF_OPTIONS + j] = value
        features.append(f)
    return features


def choose_endgame_bluff_action(state):
    """
    Decide whether to falsely claim an identity (village/wolf/madman)
    RIGHT NOW, or stay silent (only ever asked while alive_count is 3
    or 4, and this Villager hasn't already claimed anything -- see
    main.py's villager_endgame_bluff).

    Parameters
    ----------
    state : dict
        Public state (main.py's build_public_state) plus
        "self_index" and "alive_count".

    Returns
    -------
    str
        One of ENDGAME_BLUFF_OPTIONS
    """
    feature_list = _endgame_bluff_feature_list(state)
    active_policy = endgame_bluff_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, ENDGAME_BLUFF_OPTIONS)
    return chosen


def update(kind, state, action, reward, learning_rate=0.01):

    if kind == "endgame_bluff":

        feature_list = _endgame_bluff_feature_list(state)
        active_policy = endgame_bluff_split.active(state.get("day1_config"))
        active_policy.update(
            feature_list, ENDGAME_BLUFF_OPTIONS, action, reward, learning_rate
        )
        endgame_bluff_split.record_sample(state.get("day1_config"))
