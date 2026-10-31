import policy_utils
import village_ai
import board_split


# ============================================================
# Medium AI
#
# NOTE: this module used to also own the medium's OWN "reveal now
# or stay silent" timing choice (co_policy/choose_co_action, a
# 4-feature linear policy). That decision was cut from the game
# entirely once main.py's medium_day_co started forcing the Day-1
# reveal unconditionally (board[i]="Ps" with no branch) -- so
# choose_co_action had exactly zero call sites and its weights sat
# at their zero-init forever. Removed as part of a cleanup pass
# (2026) before a larger structural change; the git history (or an
# earlier delivered zip) has the code back if this timing decision
# is ever reinstated as a real choice.
# ============================================================


def update(kind, state, action, reward, learning_rate=0.01):
    """
    Apply one REINFORCE update for a logged medium decision.
    """

    if kind == "nomination_strategy":
        feature_list = [
            [1.0 if i == j else 0.0 for i in range(len(NOMINATION_OPTIONS))]
            for j in range(len(NOMINATION_OPTIONS))
        ]
        active_policy = nomination_split.active(state.get("day1_config"))
        active_policy.update(
            feature_list, NOMINATION_OPTIONS, action, reward, learning_rate
        )
        nomination_split.record_sample(state.get("day1_config"))

    elif kind == "nomination_target":
        # Same fix as village_ai.update's "execution" branch (see its
        # detailed comment): choose_nomination_target actually samples
        # from village_ai.apply_protection_floor(state, candidates), not
        # from the raw candidate list, so training must reconstruct that
        # exact same narrowed pool -- training on the unfiltered list
        # computes the REINFORCE baseline over a different distribution
        # than the one `action` was really drawn from.
        candidates = state["candidates"]
        pool = village_ai.apply_protection_floor(state, candidates)
        feature_list = [village_ai._extract_features(state, c) for c in pool]
        active_policy = nomination_target_split.active(state.get("day1_config"))
        active_policy.update(
            feature_list, pool, action, reward, learning_rate
        )
        nomination_target_split.record_sample(state.get("day1_config"))


# ============================================================
# Day 1 nomination strategy ("灰指定"): does naming a gray suspect
# right after revealing actually help, or is it better to just
# reveal and say nothing? (It can flush out the Hunter early, at
# the cost of tipping wolves off to who's being scrutinized.)
# ============================================================

NOMINATION_OPTIONS = ["nominate", "no_nomination"]

nomination_policy = policy_utils.LinearPolicy(len(NOMINATION_OPTIONS))

# Per-Day-1-board-config weight sets (2026-10 "split everything"
# extension -- see board_split.py's module docstring). Even the Day-1
# nomination is made by make_all_nominations AFTER every side's Day-1
# CO decision has already landed (see main.py's own comment on that
# call order), so day1_config is always the final, settled value here
# -- plain state.get("day1_config") is safe, no safe_day1_config()
# guard needed.
nomination_split = board_split.BoardSplitManager(nomination_policy)


def choose_nomination_strategy(state):
    """
    Decide whether to publicly name a gray suspect right after
    revealing on Day 1.

    Returns
    -------
    str
        One of "nominate", "no_nomination"
    """
    feature_list = [
        [1.0 if i == j else 0.0 for i in range(len(NOMINATION_OPTIONS))]
        for j in range(len(NOMINATION_OPTIONS))
    ]
    active_policy = nomination_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, NOMINATION_OPTIONS)
    return chosen


# ============================================================
# WHO to nominate, when choose_nomination_strategy picked
# "nominate". This USED to be a plain random.choice(gray) in
# main.py, on the reasoning that the Medium's real ability only
# ever concerns players who are ALREADY dead, so there's no
# principled evidence to point a Day-1 nomination at.
#
# That reasoning is true in the narrow sense that a lone
# nomination decision has no NEW information of its own -- but it
# ignores everything ELSE already visible on the public board by
# the time a nomination happens (CO claims, guard-verification
# results, the wolf-counting deduction, suspicious non-voters,
# etc.). A pure coin flip throws that all away for no reason. This
# reuses village_ai's own per-candidate suspicion features (the
# exact same "how suspicious does candidate X currently look"
# question the execution vote asks) so a nomination can point at
# whoever the shared body of public evidence already makes look
# worst, instead of a name pulled uniformly at random.
#
# A SEPARATE weight vector from village_ai.execution_policy is used
# (not the same one) -- nominating carries different incentives
# than voting (e.g. tipping off wolves to who's under scrutiny, or
# deliberately trying to flush out the Hunter) even though it reads
# the same underlying features.
#
# The candidate pool is NOT limited to fully-unclaimed ("gray")
# players. When another player has ALSO claimed Medium, the single
# most natural target is often that rival claimant -- exactly one of
# a contested pair has to be lying, so pointing at them can be more
# informative than pointing at someone with no claim at all. main.py
# passes every living player except the nominator as a candidate, and
# leaves it to this learned policy (via village_ai._extract_features,
# which already scores a contested co-claimant differently from a
# silent player) to decide who actually looks worst.
# ============================================================

nomination_target_policy = policy_utils.LinearPolicy(village_ai.NUM_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 extension). Same timing
# argument as nomination_split above -- always asked after Day 1's CO
# sequence has fully settled.
nomination_target_split = board_split.BoardSplitManager(nomination_target_policy)


def choose_nomination_target(state, candidates):
    """
    Decide WHO to name as the "指定" suspect, among all living players
    other than the nominator themselves -- this can include a rival
    co-claimant (e.g. another Medium claimant), not just players who
    haven't claimed anything.

    Parameters
    ----------
    state : dict
        Same public state build_public_state(day) always returns.
    candidates : list[int]
        Living players eligible to be nominated (main.py's list,
        already excluding the nominator; no longer restricted to
        unclaimed players).

    Returns
    -------
    int
        Index of the player to nominate.
    """
    # Apply the same structural-protection floor village_ai's own
    # execution vote uses (see apply_protection_floor's docstring):
    # a sole/uncontested claimant, an independently confirmed
    # villager, or the confirmed real Hunter shouldn't be a live
    # option for "who looks worst" just because a still-small learned
    # weight hasn't pushed their probability all the way to zero yet.
    # Nominating the trusted sole Hunter claimant makes no sense for a
    # genuine Medium to ever do, and without this floor it happened
    # often enough in practice (~9% of nominations) to be a real,
    # not-just-theoretical problem. Mutating state["candidates"] here
    # (main.py logs this SAME dict right after this call returns)
    # keeps choose and update looking at the identical candidate pool.
    pool = village_ai.apply_protection_floor(state, candidates)
    state["candidates"] = pool

    feature_list = [village_ai._extract_features(state, c) for c in pool]
    active_policy = nomination_target_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, pool)
    return chosen
