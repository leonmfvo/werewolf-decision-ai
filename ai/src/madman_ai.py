import policy_utils
import village_ai
import wolf_ai
import board_split


# ============================================================
# Madman AI (狂人)
#
# STEP 1 gave the Madman exactly one decision: who to vote for. STEP
# 2 (village_ai.py) downgraded several of the village's "only wolves
# ever fake-claim" hard rules to learned probabilities, because this
# step was always coming. STEP 3 gave the Madman its fake-claim
# ability: it can fake-claim Ps/Hn/Sy from Day 1, exactly like a wolf
# can, and produce the same fabricated daily reports a fake Medium/
# Seer claimant needs to keep up appearances. STEP 4 (this update)
# gives the Madman its endgame "捨てCO" (self-outing) reveal -- see
# this module's "STEP 4" section further down for the full mechanic
# and the village-side "村騙り" countermeasure it's paired with
# (villager_ai.choose_endgame_bluff_action).
#
# WHY THIS REUSES wolf_ai's MACHINERY WHEREVER POSSIBLE: the Madman
# has no private wolf-knowledge at all (confirmed by the user), so
# every wolf_ai decision that ALREADY only looks at public
# information (the day-by-day CO choice, and the fake-Medium-report
# agree/contradict choice) can be reused feature-for-feature -- only
# the reward sign and the weights differ, and both of those already
# come from having a SEPARATE LinearPolicy object per decision here,
# the same pattern vote_policy already established in Step 1. The
# one place this can't just reuse wolf_ai outright is the fake Seer
# report: wolf_ai's version includes a "vouch_teammate_white" option
# that depends on knowing who a wolf's own teammate is -- information
# the Madman doesn't have -- so that one option is dropped and the
# Madman gets its own, shorter option list and its own policy object
# instead (see MADMAN_FAKE_SEER_REPORT_OPTIONS below). The fake-
# Hunter-guard "who to falsely vouch for" declaration ALSO named a
# wolf's own teammate as one of its two options; for the Madman that
# collapses to a single meaningful choice (a random other living
# player), so it isn't worth its own LinearPolicy at all -- see
# main.py's _apply_fake_claim.
# ============================================================

vote_policy = policy_utils.LinearPolicy(village_ai.NUM_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 "split everything"
# extension -- see board_split.py's module docstring). Only ever
# happens during the day vote, strictly after Day 1's CO sequence --
# plain state.get("day1_config") is safe.
vote_split = board_split.BoardSplitManager(vote_policy)


def choose_vote_target(state):
    """
    Decide who to vote for tonight -- same candidate pool and same
    feature vector as village_ai.choose_execution_target, just scored
    by this module's own (Werewolf-aligned) weights instead.

    Parameters
    ----------
    state : dict
        Same shape as village_ai.choose_execution_target's state --
        built by main.py's build_village_state (the Madman has no
        private information to add on top of it).
    """

    candidates = state["candidates"]
    feature_list = [village_ai._extract_features(state, c) for c in candidates]

    active_policy = vote_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, candidates)

    return chosen



# ============================================================
# Day-by-day CO decision: claim one of config.CLAIMABLE_ROLES, or
# stay silent. IDENTICAL feature shape to wolf_ai.choose_co_action
# (wolf_ai.CO_OPTIONS / wolf_ai._co_feature_list) -- reused directly,
# not reimplemented, since every input it looks at (which roles are
# already claimed, whether an emergency reveal just happened, whether
# THIS player is personally under threat/nominated) is public or
# self-only information, exactly as available to the Madman as to a
# wolf. Only the weights differ: this is a SEPARATE LinearPolicy
# object, trained on the Madman's own (Werewolf-aligned) reward, so
# it can learn its own timing -- e.g. whether contesting a role the
# real Seer/Hunter already claimed is worth it for a player with no
# teammate to protect by doing so.
# ============================================================

co_policy = policy_utils.LinearPolicy(wolf_ai.NUM_CO_FEATURES)

# NOT split by board config (2026-10 user decision -- see wolf_ai.
# co_policy's matching comment): this is a reveal/stay-silent timing
# decision, one of the things that PRODUCES the Day-1 board
# configuration, so splitting it BY that configuration would be
# backwards. Stays a single shared policy.


def choose_co_action(state):
    """
    Decide whether to claim Ps/Hn/Sy, or stay silent, TODAY -- same
    options and same feature vector as wolf_ai.choose_co_action (see
    main.py's madman_day_co and the emergency-evasion check in
    progression_ai for the two moments this gets asked from).
    """
    feature_list = wolf_ai._co_feature_list(state)
    chosen, _probs = co_policy.choose(feature_list, wolf_ai.CO_OPTIONS)
    return chosen


# ============================================================
# Fake Medium report strategy (agree / contradict) -- IDENTICAL
# option set and feature shape to wolf_ai.choose_fake_report_strategy,
# reused for the same reason as the CO decision above: this choice
# never depended on knowing a teammate's identity in the first place,
# only on the real Medium's already-public verdict.
# ============================================================

fake_report_policy = policy_utils.LinearPolicy(wolf_ai.NUM_FAKE_REPORT_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 extension). Only ever
# asked the day after an execution -- well past Day 1's CO sequence --
# plain state.get("day1_config") is safe.
fake_report_split = board_split.BoardSplitManager(fake_report_policy)


def choose_fake_report_strategy(state):
    """
    Decide whether a fake Medium claim's daily report should agree
    with or contradict the real Medium's verdict on yesterday's
    execution. See wolf_ai.choose_fake_report_strategy -- identical
    shape, separate (Werewolf-aligned) weights.
    """
    feature_list = wolf_ai.fake_report_feature_list(state)
    active_policy = fake_report_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, wolf_ai.FAKE_REPORT_OPTIONS)
    return chosen


# ============================================================
# Fake Seer report strategy. UNLIKE the two decisions above, this
# can't just reuse wolf_ai's option list: wolf_ai.FAKE_SEER_REPORT_
# OPTIONS includes "vouch_teammate_white", which depends on knowing a
# wolf teammate's identity -- something the Madman never has. Dropping
# that one option leaves three, all of them already public-information
# choices: frame a rival Seer claimant, or play it safe with a random
# accusation/clearance.
# ============================================================

MADMAN_FAKE_SEER_REPORT_OPTIONS = ["accuse_rival_black", "random_black", "random_white"]

# DAY 1 ONLY: see wolf_ai.FAKE_SEER_REPORT_DAY1_OPTIONS's comment for
# the full reasoning -- a claimant's first-ever public report always
# stands in for the mandatory pre-game "random white" investigation
# (main.py's _assign_seer_random_white), which the real Seer can NEVER
# report as black. With no teammate to vouch for, the Madman's white
# option collapses to a single choice: "random_white" is the only
# entry left once the two black options are excluded, so this isn't
# even a live decision on Day 1 any more (LinearPolicy.update's
# gradient against a single-candidate set is naturally zero, so no
# special-casing is needed to keep training inert about it).
MADMAN_FAKE_SEER_REPORT_DAY1_OPTIONS = ["random_white"]

fake_seer_report_policy = policy_utils.LinearPolicy(len(MADMAN_FAKE_SEER_REPORT_OPTIONS))

# Per-Day-1-board-config weight sets (2026-10 extension). Even the
# Day-1 instance is reported alongside the claim itself, after Day 1's
# whole CO sequence has settled (see wolf_ai.fake_seer_report_split's
# matching comment) -- plain state.get("day1_config") is safe.
fake_seer_report_split = board_split.BoardSplitManager(fake_seer_report_policy)


def choose_fake_seer_report_strategy(state):
    """
    Decide what today's fabricated Seer "investigation result" should
    be -- see main.py's fake_seer_report for how each option resolves
    to an actual (target, is_wolf) pair. No teammate-vouching option
    (see this module's docstring for why).

    On Day 1 the candidate list is narrowed to MADMAN_FAKE_SEER_
    REPORT_DAY1_OPTIONS (a black verdict is structurally impossible
    that day -- see its comment), the same way wolf_ai.choose_fake_
    seer_report_strategy narrows its own option list.
    """
    options = (
        MADMAN_FAKE_SEER_REPORT_DAY1_OPTIONS if state.get("day") == 1
        else MADMAN_FAKE_SEER_REPORT_OPTIONS
    )
    option_indices = [MADMAN_FAKE_SEER_REPORT_OPTIONS.index(o) for o in options]
    feature_list = [
        [1.0 if i == idx else 0.0 for i in range(len(MADMAN_FAKE_SEER_REPORT_OPTIONS))]
        for idx in option_indices
    ]
    active_policy = fake_seer_report_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, options)
    return chosen


# ============================================================
# "accuse_rival_black" TARGET selection -- see wolf_ai's comment on
# choose_accuse_target for the full reasoning (this reuses wolf_ai's
# _accuse_target_feature_list directly: the Madman has no teammates to
# exclude, but "which non-teammate rival is worth accusing" is the
# exact same question with the exact same available public
# information, so it earns its own separate weight vector rather than
# its own separate feature design).
# ============================================================

accuse_target_policy = policy_utils.LinearPolicy(wolf_ai.NUM_ACCUSE_TARGET_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 extension). Reached only
# from within fake_seer_report_strategy's safe-to-split "accuse_rival_
# black" branch -- plain state.get("day1_config") is safe.
accuse_target_split = board_split.BoardSplitManager(accuse_target_policy)


def choose_accuse_target(state):
    """
    Pick WHICH candidate in state["candidates"] to name in an
    "accuse_rival_black" fake Seer report. Only ever called when
    len(candidates) > 1 (see wolf_ai.choose_accuse_target's docstring).
    """
    feature_list = wolf_ai._accuse_target_feature_list(state)
    active_policy = accuse_target_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, state["candidates"])
    return chosen


# ============================================================
# STEP 4: the endgame "捨てCO" (self-outing) reveal.
#
# Once only 3 or 4 players remain, the Madman can drop its pretense
# and openly declare itself hostile to the village -- config.py's
# comment on "Kg" and the user's own worked example explain WHY this
# helps: with so few players left, the Madman's own vote and a wolf's
# vote landing on DIFFERENT targets (because neither can tell the
# other apart from the real villager under the fog of public
# information alone) can accidentally split the vote away from the
# one player who actually needs to be executed for the wolves to win.
# An open declaration removes that fog for the remaining wolves (who
# now know exactly who NOT to vote for), at the cost of also removing
# it for the village (who now know exactly who TO vote for). Whether
# that trade is worth it is exactly what training has to discover --
# hence a real learned decision, not a hard-coded rule.
#
# THE REVEAL ITSELF is recorded in main.py's self_declared_non_village
# list, NOT in co_history/board the way a Ps/Hn/Sy claim is -- it's a
# different kind of public fact ("I am not on the village's side" as
# opposed to "I am the Medium"), and -- critically -- it's the exact
# same list a genuine Villager's endgame "村騙り" bluff (villager_ai.
# choose_endgame_bluff_action) writes into. Sharing one list is what
# actually MAKES the village-side bluff work as a countermeasure: an
# outside observer (or a feature built from this list) cannot tell
# whether a given entry came from the real Madman or from a bluffing
# Villager just by looking at the list's shape.
# ============================================================

# Four options, not two: instead of a single "reveal" meaning "claim
# hostile", the Madman can now claim any one of the three team
# identities specifically -- "狂人CO" (claim_madman, its own true
# identity -- the ORIGINAL "reveal" option this replaced), "狼CO"
# (claim_wolf, borrowing the wolves' cover instead of its own), or
# "村人CO" (claim_village, the subtler reassurance play). See
# main.py's endgame_claims / wolf_ai.ENDGAME_DECLARE_OPTIONS's comment
# for how a contested "狂人CO" claim (at most one real Madman exists)
# becomes a hard deduction the same way a contested Ps/Hn/Sy claim
# already is.
ENDGAME_REVEAL_OPTIONS = ["stay_hidden", "claim_village", "claim_wolf", "claim_madman"]

# Context values interacted per option, mirroring seer_ai._co_feature_
# list's exact shape (one-hot option identity + context*option
# interaction terms) -- NOT a single shared feature vector scored the
# same way for every option, which would make every option always
# score identically and leave nothing for training to ever learn. See
# village_ai.endgame_context_features's docstring for what each
# context value means.
NUM_ENDGAME_REVEAL_OPTIONS = len(ENDGAME_REVEAL_OPTIONS)
NUM_ENDGAME_REVEAL_FEATURES = (
    NUM_ENDGAME_REVEAL_OPTIONS
    + NUM_ENDGAME_REVEAL_OPTIONS * village_ai.NUM_ENDGAME_CONTEXT_FEATURES
)

endgame_reveal_policy = policy_utils.LinearPolicy(NUM_ENDGAME_REVEAL_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 extension). Only ever
# asked once alive_count is 3-4, long after Day 1 -- plain
# state.get("day1_config") is safe.
endgame_reveal_split = board_split.BoardSplitManager(endgame_reveal_policy)


def _endgame_reveal_feature_list(state):
    context = village_ai.endgame_context_features(state)
    features = []
    for j in range(NUM_ENDGAME_REVEAL_OPTIONS):
        f = [0.0] * NUM_ENDGAME_REVEAL_FEATURES
        f[j] = 1.0
        for k, value in enumerate(context):
            f[NUM_ENDGAME_REVEAL_OPTIONS + k * NUM_ENDGAME_REVEAL_OPTIONS + j] = value
        features.append(f)
    return features


def choose_endgame_reveal_action(state):
    """
    Decide whether to openly claim an identity (village/wolf/madman)
    RIGHT NOW, or stay hidden (only ever asked while alive_count is 3
    or 4 -- see main.py's madman_endgame_reveal).

    Parameters
    ----------
    state : dict
        Public state (main.py's build_public_state) plus
        "self_index" (this Madman's own board index, so
        already_has_role_claim can be computed) and "alive_count".

    Returns
    -------
    str
        One of ENDGAME_REVEAL_OPTIONS
    """
    feature_list = _endgame_reveal_feature_list(state)
    active_policy = endgame_reveal_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, ENDGAME_REVEAL_OPTIONS)
    return chosen


def update(kind, state, action, reward, learning_rate=0.01):
    """
    Apply one REINFORCE update for a single logged Madman decision.
    Dispatches on `kind`, mirroring wolf_ai.update's shape -- the
    Madman now has the same FAMILY of decisions a wolf does (vote,
    day-by-day CO, fake reports), just scored by its own separate
    weights and reward sign.
    """

    if kind == "vote_target":

        candidates = state["candidates"]
        feature_list = [village_ai._extract_features(state, c) for c in candidates]
        active_policy = vote_split.active(state.get("day1_config"))
        active_policy.update(feature_list, candidates, action, reward, learning_rate)
        vote_split.record_sample(state.get("day1_config"))

    elif kind == "co_action":

        feature_list = wolf_ai._co_feature_list(state)
        co_policy.update(feature_list, wolf_ai.CO_OPTIONS, action, reward, learning_rate)

    elif kind == "fake_report_strategy":

        feature_list = wolf_ai.fake_report_feature_list(state)
        active_policy = fake_report_split.active(state.get("day1_config"))
        active_policy.update(feature_list, wolf_ai.FAKE_REPORT_OPTIONS, action, reward, learning_rate)
        fake_report_split.record_sample(state.get("day1_config"))

    elif kind == "fake_seer_report_strategy":

        # Must mirror choose_fake_seer_report_strategy's exact
        # candidate set for the day (see wolf_ai.update's matching
        # comment on why a mismatched set gives a wrong baseline).
        options = (
            MADMAN_FAKE_SEER_REPORT_DAY1_OPTIONS if state.get("day") == 1
            else MADMAN_FAKE_SEER_REPORT_OPTIONS
        )
        option_indices = [MADMAN_FAKE_SEER_REPORT_OPTIONS.index(o) for o in options]
        feature_list = [
            [1.0 if i == idx else 0.0 for i in range(len(MADMAN_FAKE_SEER_REPORT_OPTIONS))]
            for idx in option_indices
        ]
        active_policy = fake_seer_report_split.active(state.get("day1_config"))
        active_policy.update(feature_list, options, action, reward, learning_rate)
        fake_seer_report_split.record_sample(state.get("day1_config"))

    elif kind == "fake_seer_accuse_target":

        candidates = state["candidates"]
        feature_list = wolf_ai._accuse_target_feature_list(state)
        active_policy = accuse_target_split.active(state.get("day1_config"))
        active_policy.update(feature_list, candidates, action, reward, learning_rate)
        accuse_target_split.record_sample(state.get("day1_config"))

    elif kind == "endgame_reveal":

        feature_list = _endgame_reveal_feature_list(state)
        active_policy = endgame_reveal_split.active(state.get("day1_config"))
        active_policy.update(feature_list, ENDGAME_REVEAL_OPTIONS, action, reward, learning_rate)
        endgame_reveal_split.record_sample(state.get("day1_config"))
