import math
import itertools

import policy_utils
import config
import board_split

CLAIMABLE_ROLES = config.CLAIMABLE_ROLES


# ============================================================
# Village AI
#
# Decides who to execute during the day, using ONLY public
# information (the village has no private knowledge of anyone's
# true role).
#
# GENERALIZATION: the claim-tracking features (is claimant / is
# contested / is sole claimant / is provably fake) are built by
# LOOPING over config.CLAIMABLE_ROLES, instead of being hard-coded
# per role name. Adding a new claimable role (e.g. a Seer) to
# config.py automatically gives it the same set of 4 features here
# -- no code changes needed in this file for THAT part. What is
# still role-specific (because it depends on what that role's
# ability actually IS) is the guard-instruction verification
# (is_confirmed_fake_hunter / is_confirmed_real_hunter) -- a
# hypothetical new role with a different ability would need its
# own equivalent verification feature, the same way this one was
# custom-built for the Hunter's guard.
# ============================================================

NUM_ROLE_FEATURES = 6  # is_claimant, is_claimant*count, is_sole,
                        # is_provably_fake, contest_wolf_probability,
                        # is_lone_survivor_of_contest (see its own
                        # comment on _build_feature_vector's role loop)

# [0] bias
# [1 .. 6*len(CLAIMABLE_ROLES)] : 6 features per claimable role, in
#     config.CLAIMABLE_ROLES order
# then: is_villager_claimant, is_confirmed_villager,
#       is_confirmed_fake_hunter, is_confirmed_real_hunter,
#       alive_count (normalized), guard_success_count (normalized),
#       is_falsely_vouched, is_pending_verification,
#       is_nominated_by_sole_claimant, is_nominated_by_contested_claimant,
#       is_impossible_late_forced_claim, is_confirmed_contradicting_medium,
#       is_last_wolf_pressure
#       is_vouched_for_executed,
#       suspicious_non_vote_count (normalized),
#       is_nominated_by_vindicated_contested_claimant,
#       is_this_voters_own_nomination,
#       defied_sole_nomination_count (normalized),
#       voted_for_protected_count (normalized),
#       is_accused_by_sole_seer, is_accused_by_contested_seer,
#       is_confirmed_contradicting_seer, is_accused_by_confirmed_real_seer,
#       is_self_declared_non_village, is_this_voters_own_seer_accusation,
#       is_kg_claimant, is_sole_kg_claimant, contest_kg_wolf_probability,
#       is_accused_by_track_recorded_seer, is_confirmed_wolf_by_real_seer,
#       off_line_vote_count (normalized),
#       same_role_wolf_found, is_last_wolf_pressure_by_claim
#   --- 2026-10-06 additions (appended; see _extra_features) ---
#       voted_for_confirmed_white_count, voted_for_confirmed_wolf_count,
#       is_whited_by_sole_seer, is_whited_by_contested_seer,
#       is_whited_by_confirmed_real_seer, is_unreported_gray,
#       seer_medium_conflict_count, false_result_count,
#       is_my_same_role_rival, is_own_white, is_own_black,
#       late*is_claimant, late*is_sole_claimant, late*is_gray,
#       final*is_claimant, final*is_gray
#       (second batch) sy_all_agree_white/black, is_split_target,
#       black/white_from_proven_fake_seer, rival_ps_black/white,
#       tight*{sole,confirmed_villager,gray,claimant}, loose*claimant,
#       no_hn*gray, gray_ratio*unreported_gray, private_contradiction,
#       nominator_inconsistency, not_voting_own_black, voted_claimed_white
NUM_BASE_TAIL_FEATURES = 33
NUM_EXTRA_TAIL_FEATURES = 34  # 2026-10-06 additions, see _extra_features (16 + 18 added later the same day)
NUM_FIXED_TAIL_FEATURES = NUM_BASE_TAIL_FEATURES + NUM_EXTRA_TAIL_FEATURES

NUM_FEATURES = 1 + NUM_ROLE_FEATURES * len(CLAIMABLE_ROLES) + NUM_FIXED_TAIL_FEATURES

# ------------------------------------------------------------
# Some of the features above aren't ordinary probabilistic suspicion
# signals -- they're STRUCTURAL CERTAINTIES, deduced facts that are
# true or false with (in-simulation) 100% confidence: a guard
# verification proving a Hunter claim fake/real, a claim-order
# contradiction proving a claim fake, a Medium claim appearing on a
# day the real Medium could never have stayed silent through, or a
# Medium's own past report directly contradicting the real Medium's.
# Left at the SAME learning-rate/decay-rate as every ordinary,
# noise-prone suspicion feature, these never grow past a small
# fraction of the weight they deserve -- they get outweighed by
# comparatively weak-but-common priors like "trust a sole claimant"
# (see the ATTACK_LR_MULTIPLIER comment in wolf_ai.py for the first
# time this exact tension was found, there for the wolf's "bite the
# known Hunter" decision). Boosting just these indices lets a proven
# fact commit toward the extreme value it should have, without
# touching how fast the genuinely uncertain features nearby learn.
#
# Feature index layout (see _build_feature_vector below). These are
# computed FROM the current NUM_ROLE_FEATURES/CLAIMABLE_ROLES/tail
# layout rather than hard-coded absolute numbers -- the previous
# version hard-coded [10, 11, 12, 19, 20], which was only correct for
# EXACTLY 2 claimable roles (Ps, Hn). Adding a 3rd claimable role
# (Sy) shifts every tail feature's absolute index by
# NUM_ROLE_FEATURES=4, so those hard-coded numbers would have quietly
# started boosting the LR/decay of the WRONG features (alive_count,
# guard_count, ...) while leaving the real "confirmed" features at
# the ordinary rate -- a latent bug that only became visible once the
# role count actually changed. Recomputing from the tail's known,
# fixed relative offsets keeps this correct regardless of how many
# claimable roles config.py defines.
#   1 + NUM_ROLE_FEATURES*i + 3 : is_provably_fake for CLAIMABLE_ROLES[i]
#                                 (claim-order contradiction -- structurally certain)
#   tail offset 1  : is_confirmed_villager
#   tail offset 2  : is_confirmed_fake_hunter
#   tail offset 3  : is_confirmed_real_hunter
#   tail offset 10 : is_impossible_late_forced_claim
#   tail offset 11 : is_confirmed_contradicting_medium
#   tail offset 21 : is_confirmed_contradicting_seer
#   tail offset 24 : is_this_voters_own_seer_accusation (see its own
#                     NOTE below for why this one is structurally
#                     certain too, despite being added much later)
#
# NOTE (Madman/"Kg" Step 2): is_accused_by_confirmed_real_seer (tail
# offset 22) is DELIBERATELY NOT in this boosted list, even though it
# replaces what used to be an unconditional hard-rule execute. Once a
# Madman can also fake-claim Sy (a later step), "this Sy claimant is
# in confirmed_villagers" no longer proves they're the genuine Seer
# with 100% certainty -- confirmed_villagers only proves "not
# literally a Wo", and a Madman satisfies that trivially while still
# possibly lying about what they saw. So this is now a STRONG but
# genuinely uncertain prior, learned at the same ordinary rate as
# every other probabilistic suspicion signal -- not boosted like the
# claim-fakeness proofs above, which stay 100% certain regardless of
# Madman (only WHICH team the confirmed-fake claimant belongs to
# becomes uncertain, not whether their claim itself is fake).
#
# NOTE (Madman/"Kg" Steps 4-5): is_self_declared_non_village (tail
# offset 23) is ALSO deliberately excluded, for a stronger reason than
# is_accused_by_confirmed_real_seer above -- it isn't even a strong
# prior. A genuine Villager's endgame "村騙り" bluff (villager_ai.
# choose_endgame_bluff_action) writes into the exact same public list
# a genuine Madman reveal does specifically so this signal CAN'T be
# trusted as a proof, by design (see that function's docstring). This
# stays an ordinary, ambiguous, fully-learned suspicion feature.
#
# NOTE (own-seer-accusation self-trust): is_this_voters_own_seer_
# accusation (tail offset 24) IS included in this boosted list, even
# though it was added long after the Madman steps above and looks at
# first glance like it should be an ordinary suspicion feature the
# way is_this_voters_own_nomination is. It isn't: run_vote (main.py)
# routes EVERY wolf voter through wolf_ai.choose_vote_target and
# EVERY Madman voter through madman_ai.choose_vote_target, no matter
# what role either one has fake-claimed -- only a voter whose real
# role is genuinely "Sy" ever reaches THIS policy while also holding
# a seer_reports entry with claimant == themselves. So whenever this
# feature is true, the voter is, with total certainty (not merely "a
# claimant found non-wolf by some independent means", the weaker
# is_accused_by_confirmed_real_seer case above), the one real Seer,
# reporting their own genuine investigation. There's no equivalent
# certainty for is_this_voters_own_nomination, which is why THAT one
# stays at the ordinary rate: a Medium's nomination is public and
# could in principle be examined by any village-side voter, whereas
# this is the voter's own private ground truth about themselves.
#
# NOTE (2026-10-04 "ABC elimination" addition): is_confirmed_wolf_
# by_real_seer (tail offset 29) IS included in this boosted list,
# unlike is_accused_by_confirmed_real_seer (offset 22) above. The
# difference is exactly the one that NOTE explains: offset 22 is fed
# by ANY confirmed-non-wolf Sy claimant's accusation (which could be a
# bluffing Madman's fabricated report), so it stays an ordinary,
# uncertain, un-boosted signal. Offset 29 is fed ONLY from
# state["confirmed_wolves"] (main.py's build_public_state), which is
# populated ONLY from a Sy claimant found in confirmed_real_claimants
# -- the airtight, identity-proven pigeonhole case, not mere non-wolf
# status (see _is_confirmed_contradicting_seer's 2026-10-04 bug-fix
# note for why that distinction matters). Once a claimant is genuinely
# confirmed to BE the real Seer, their every report is ground truth by
# the game's own rules (the ability has no noise), so a "wolf" verdict
# from them is exactly as certain as a guard-instruction-verified
# confirmed_real_hunter fact -- hence the same boosted treatment.
# ------------------------------------------------------------
CONFIRMED_LR_MULTIPLIER = 1.5
CONFIRMED_DECAY_MULTIPLIER = 0.35

_TAIL_START = 1 + NUM_ROLE_FEATURES * len(CLAIMABLE_ROLES)
_CONFIRMED_TAIL_OFFSETS = [1, 2, 3, 10, 11, 21, 24, 29, NUM_BASE_TAIL_FEATURES + 7, NUM_BASE_TAIL_FEATURES + 8]

_CONFIRMED_FEATURE_INDICES = (
    [1 + NUM_ROLE_FEATURES * i + 3 for i in range(len(CLAIMABLE_ROLES))]
    + [_TAIL_START + off for off in _CONFIRMED_TAIL_OFFSETS]
)


def _make_execution_multipliers(base_value):
    multipliers = [1.0] * NUM_FEATURES
    for idx in _CONFIRMED_FEATURE_INDICES:
        multipliers[idx] = base_value
    return multipliers


execution_policy = policy_utils.LinearPolicy(
    NUM_FEATURES,
    lr_multipliers=_make_execution_multipliers(CONFIRMED_LR_MULTIPLIER),
    decay_multipliers=_make_execution_multipliers(CONFIRMED_DECAY_MULTIPLIER),
)

# Per-Day-1-board-config weight sets for execution voting (see
# board_split.py). execution_policy above stays the shared/common
# policy exactly as before -- every existing reader of its .weights
# (train_and_report.py's reports, persistence.py's save/load) is
# unaffected. execution_split.active(day1_config) is what actually
# decides/learns from this point on; it's execution_policy itself
# until train_and_report.enable_board_split() is called.
execution_split = board_split.BoardSplitManager(execution_policy)


def _is_provably_fake(state, candidate, own_co):
    """
    True if an EARLIER claimant (by claim day) of the SAME role
    has since died -- there's only one true holder of each role,
    so a claim appearing after the earlier one's death is almost
    certainly fake.
    """
    if own_co not in ("Ps", "Sy"):
        # 2026-10-06 (round 4): only roles whose Day-1 CO is forced.
        # A Hunter may reveal late (after a fake Hn died), so "an
        # earlier Hn claimant is dead" proves nothing about a later one.
        return False

    claim_day = state.get("claim_day")
    if not claim_day:
        return False

    my_day = claim_day[candidate]
    if my_day is None:
        return False

    co_history = state["co_history"]
    board = state["board"]

    for j, co in enumerate(co_history):
        if j == candidate or co != own_co:
            continue
        other_day = claim_day[j]
        if other_day is not None and other_day < my_day and board[j] == "--":
            return True

    return False


def _is_confirmed_fake_by_cleared_rival(state, candidate, own_co):
    """
    True if the SAME role has another claimant who has been
    independently confirmed non-wolf (state["confirmed_villagers"] --
    e.g. a night-attack victim, or a guard-verified survivor), while
    THIS candidate has NOT.

    Only one real holder of each claimable role exists, and in the
    current ruleset ONLY wolves ever fake-claim a role (main.py's
    villager_day_co, the option for a genuine Villager to bluff a
    claim, is disabled -- see the NOTE by that call in day_phase). So
    the moment any one claimant of a contested role is cleared as
    genuinely non-wolf by some UNRELATED deduction, every other
    claimant of that same role is a wolf with total certainty --
    regardless of claim order, and regardless of whether the cleared
    claimant is still alive. This is what actually lets something
    like "the other Medium claimant just got attacked and died, so
    THIS one must be lying" be recognized immediately, instead of
    only ever catching a claim proven fake by the older claim-order
    rule (_is_provably_fake), which requires an EARLIER same-role
    claimant to have died and says nothing about same-day contests.

    2026-10-04 REVERTED a same-day attempt to tighten this to
    state["confirmed_real_claimants"] (the pigeonhole-only, fully
    IDENTITY-proven list) instead of confirmed_villagers, to close the
    theoretical Madman-trap gap documented in _is_confirmed_
    contradicting_seer's history. Measured empirically (a live training
    run) that this was a large NET REGRESSION: the trap requires the
    Madman to BOTH claim this exact contested role AND be the specific
    claimant who happens to get confirmed non-wolf while the OTHER
    claimant is the true holder -- a narrow coincidence -- whereas the
    ordinary case this rule exists for (a role's claim resolved the
    instant ONE claimant dies to a night attack, Madman or not) is
    extremely common and was being thrown away along with it, visibly
    crashing village win rate (measured ~50-60% -> ~7-10% and NOT
    recovering across several hundred further epochs, unlike ordinary
    co-evolution noise). The real fix for the narrow edge case is the
    separate, ADDITIVE is_confirmed_wolf_by_real_seer feature (fed only
    from the airtight confirmed_real_claimants/confirmed_wolves data),
    not replacing this already-usually-correct heuristic.
    """
    if own_co not in CLAIMABLE_ROLES:
        return False

    confirmed_villagers = state.get("confirmed_villagers", [])
    if candidate in confirmed_villagers:
        return False

    co_history = state["co_history"]
    for j, co in enumerate(co_history):
        if j != candidate and co == own_co and j in confirmed_villagers:
            return True

    return False


def _is_self_contradicting_seer_claim(state, candidate, own_co):
    """
    True if `candidate` claims "Sy" and their OWN public reports give
    two different verdicts for the SAME target on different days
    (e.g. "P5は白でした" one day, "P5は黒でした" a later day). A real
    Seer's result for a fixed target never changes -- alignment is
    fixed for the whole game -- so this is an unconditional,
    self-contained proof the claim is fake. Unlike
    _is_confirmed_contradicting_seer (which needs some OTHER Sy
    claimant to already be independently confirmed the real Seer
    before it can fire), this needs no outside confirmation at all:
    the claimant convicts themselves. Folded into is_provably_fake
    alongside the other two fakeness proofs so it reuses the existing
    per-role boosted tail feature (no new feature slot needed).
    """
    if own_co != "Sy":
        return False

    verdict_by_target = {}
    for r in state.get("seer_reports", []):
        if r["claimant"] != candidate:
            continue
        target = r["player"]
        if target in verdict_by_target and verdict_by_target[target] != r["is_wolf"]:
            return True
        verdict_by_target[target] = r["is_wolf"]

    return False


def _build_feature_vector(own_co, claim_counts, is_villager_claim,
                           is_confirmed_villager, is_confirmed_fake_hunter,
                           is_confirmed_real_hunter, alive_count, guard_count,
                           alive_claim_counts=None,
                           provably_fake_role=None, is_falsely_vouched=False,
                           is_pending_verification=False,
                           is_nominated_by_sole=False, is_nominated_by_contested=False,
                           is_impossible_late_forced_claim=False,
                           is_confirmed_contradicting_medium=False,
                           is_last_wolf_pressure=False,
                           is_vouched_for_executed=False,
                           suspicious_non_vote_count=0,
                           is_nominated_by_vindicated_contested_claimant=False,
                           is_this_voters_own_nomination=False,
                           defied_sole_nomination_count=0,
                           voted_for_protected_count=0,
                           is_accused_by_sole_seer=False,
                           is_accused_by_contested_seer=False,
                           is_confirmed_contradicting_seer=False,
                           is_accused_by_confirmed_real_seer=False,
                           is_self_declared_non_village=False,
                           is_this_voters_own_seer_accusation=False,
                           is_kg_claimant=False, is_sole_kg_claimant=False,
                           contest_kg_wolf_probability=0.0,
                           is_accused_by_track_recorded_seer=False,
                           is_confirmed_wolf_by_real_seer=False,
                           off_line_vote_count=0,
                           same_role_wolf_found=False,
                           is_last_wolf_pressure_by_claim=False,
                           extra=None):
    """
    Shared construction used by both _extract_features (live play)
    and lookup (reporting), so the two can never drift apart.

    claim_counts: dict {role: count_of_EVER claimants (co_history,
        dead included)} -- the raw pigeonhole population used by
        is_sole/contest_wolf_probability below (whether one of the
        `count` people who EVER claimed this role is genuine doesn't
        change just because one of them has since died with no
        further information revealed).
    alive_claim_counts: dict {role: count_of_CURRENTLY-ALIVE
        claimants}, used only by is_lone_survivor_of_contest below.
        Optional (defaults to None, which disables that one feature
        via the fallback in the role loop) so lookup()'s reporting
        callers that don't track alive/dead separately keep working.
    provably_fake_role: pass own_co here if this candidate's claim
        has been proven fake by the claim-order deduction; None
        otherwise.
    is_falsely_vouched: a CONFIRMED FAKE Hunter claimant lied that
        this candidate was "confirmed white" -- suspicious, since
        that liar may have been protecting a wolf teammate.
    is_pending_verification: this candidate is CURRENTLY one of a
        contested pair with a guard-instruction verification not
        yet resolved (results only come after tonight). Lets the
        village learn whether it's worth NOT executing either
        claimant yet -- letting them "swim" one more day so the
        verification can resolve who's genuine with certainty --
        rather than gambling on a 50/50 guess today.
    is_nominated_by_sole / is_nominated_by_contested: this candidate
        was publicly named ("灰指定") by a Medium claimant -- kept
        as TWO separate features (rather than one) so the village
        can learn to weigh a nomination from an UNCONTESTED (more
        likely genuine) claimant differently from one made by a
        claimant who's currently part of a contested pair (who
        might just be a wolf trying to redirect suspicion).
    is_impossible_late_forced_claim: BOTH the Medium and the Seer are
        REQUIRED to reveal on Day 1 (main.py's medium_day_co and
        seer_day_co each force this unconditionally -- matching
        standard human convention for both roles, not just the
        Medium). So a "Ps" or "Sy" claim made on any OTHER day is not
        just suspicious -- it's a STRUCTURAL IMPOSSIBILITY for the
        real Medium/Seer, since they'd have already revealed on Day 1
        if alive. This is a stronger-than-"provably fake" signal: it
        needs no comparison to another claimant at all, and it's
        shared across both forced-reveal roles (rather than split
        into is_impossible_late_medium_claim / is_impossible_late_
        seer_claim) since it means exactly the same thing for either
        one: "this specific claim could not possibly be the genuine
        holder's."
    is_confirmed_contradicting_medium: this Medium claimant's OWN
        reported verdict on some past execution directly disagrees
        with the real Medium's verdict on that SAME player -- an
        absolute, structural proof they're the fake one (only one
        report can be true).
    is_last_wolf_pressure: GLOBAL wolf-count tracking -- if all but
        ONE wolf has already been confirmed dead (via Medium
        reports), and this candidate is part of a STILL-UNRESOLVED
        contested claim (which is guaranteed to contain a wolf),
        then this candidate is where the very last wolf must be
        hiding. Combines information across DIFFERENT roles' claim
        contests (e.g. a resolved Hunter contest + an unresolved
        Medium contest) that no single-role feature above can see.
    is_vouched_for_executed: this candidate (as a fake Hunter
        claimant) publicly vouched for someone as "confirmed white"
        -- and the village went on to EXECUTE that exact person
        anyway. This doesn't require the voucher to be formally
        confirmed fake first; a genuine ally's vouching target
        getting executed by the village's own independent judgment
        is itself a red flag worth learning from.
    suspicious_non_vote_count: how many times, on a day the village
        LATER learned (via a trustworthy Medium report) that the
        person executed that day really was a wolf, this candidate
        voted for someone ELSE instead -- i.e. failed to help catch
        a wolf they had exactly the same public information to spot.
        This is only ever suggestive evidence (an honest villager
        can easily have suspected someone else that day for good
        reason), never a certainty, which is exactly why it's a
        LEARNED feature here rather than a hard rule like the
        confirmed_* facts above -- the training process gets to
        decide how much weight this kind of pattern deserves.
    is_nominated_by_vindicated_contested_claimant: this candidate
        was nominated ("灰指定") by a Medium claimant who, AT THE
        TIME, was part of a contested (2+) claim -- so ordinarily
        this only earns the weaker is_nominated_by_contested signal,
        since a contested claimant could easily be the wolf trying
        to redirect suspicion. But if that SAME nominator has SINCE
        been independently vindicated (added to confirmed_villagers
        by some other legitimate means -- e.g. the wolf-counting
        pigeonhole deduction saturating the known wolf count), their
        nomination deserves to be trusted close to how an uncontested
        claimant's nomination already is, not lumped in with every
        other contested nomination that never got vindicated. This
        is the nomination-side counterpart to suspicious_non_vote_count
        above: both are ways of letting later-earned certainty
        retroactively re-grade an earlier, murkier signal.
    is_this_voters_own_nomination: SELF-REFERENTIAL -- true only for
        the one voter who is themselves the Medium claimant that
        nominated this candidate, when evaluating whether to vote for
        their own nomination target. Every other voter always sees
        this as false for every candidate. Without this, a Medium's
        public nomination and their own vote are two completely
        independent decisions from two completely independent
        policies (nomination_policy vs. this shared execution_policy)
        with nothing tying them together -- so a Medium naming P5 as
        suspicious and then voting for P2 instead looks exactly as
        likely as voting for P5, which reads as a glaring, needless
        inconsistency. This lets the village-side policy learn
        whether following through on your own public nomination is
        actually the better play (it usually should be, since a
        genuine Medium naming someone and then voting elsewhere for
        no visible reason is exactly the kind of tell a real wolf
        pretending to be a nervous Medium might leave), rather than
        the two being uncorrelated by construction.
    defied_sole_nomination_count: how many times THIS candidate voted
        for someone OTHER than the target publicly named ("指定") by
        an, AT THE TIME, uncontested (sole) Medium claimant -- while
        that nominee was still alive to vote for. Modeled on real
        human play, where ignoring a trusted, uncontested Medium's
        read without a visible reason ("制裁吊り" territory) is itself
        read as suspicious: a genuine villager usually has no reason
        to buck a read nobody has any grounds yet to doubt, while a
        wolf has an obvious one (the nominee might be their own
        teammate, or simply not who they want executed). Unlike
        suspicious_non_vote_count (which needs a LATER-confirmed
        wolf to even compute), this needs no hindsight at all -- the
        sole Medium's nomination is trustworthy the moment it's made,
        so this can accumulate evidence in real time, day by day.
    voted_for_protected_count: how many times THIS candidate has cast
        a vote for someone apply_protection_floor would have excluded
        from the village's OWN candidate pool that same day (a sole/
        uncontested claimant, an already-confirmed villager, or the
        confirmed real Hunter). The mirror image of
        defied_sole_nomination_count: that one catches failing to
        comply with a trustworthy call, this one catches actively
        voting to kill someone that same protection floor already
        vouches for. A genuine village-aligned voter has no reason to
        do this -- their own execution policy already avoids it by
        construction -- while a wolf's separate vote policy
        (wolf_ai.vote_policy) has no equivalent structural restraint,
        only a still-imperfect learned deterrent. This lets the
        village catch that gap after the fact, the same real-time way
        defied_sole_nomination_count does.
    contest_wolf_probability (per claimable role, computed inline
        below -- not a constructor parameter, since it's fully
        derived from claim_counts/own_co already passed in): the
        EXACT probability this candidate is the wolf, GIVEN ONLY that
        they are one of `count` current claimants of this role and
        nothing else. Only one real holder of a role exists, and
        (per this ruleset) only wolves ever fake-claim a role, so
        among `count` claimants exactly 1 is genuine and `count - 1`
        are wolves -- absent any other information, this specific
        claimant is the wolf with probability (count - 1) / count.
        For count == 1 (sole claimant) this is exactly 0 (matches
        is_sole's implicit trust); for count == 2 it's 0.5 (a genuine
        coin flip -- see is_pending_verification, which exists
        precisely to let the village learn to wait rather than guess
        here); for count == 3 it's 0.667; and it keeps climbing
        toward (but never reaching) 1.0 as more and more claimants
        pile onto the same role.
        WHY THIS NEEDS ITS OWN FEATURE rather than relying on the
        existing is_claimant*count term to teach the same lesson: a
        LINEAR model can only ever learn a straight-line response to
        a raw count (0, 1, 2, 3, 4, ...), but the true target here is
        the saturating curve (count-1)/count (0, 0.5, 0.667, 0.75,
        0.8, ...) -- unbounded linear growth can be tuned to roughly
        match the curve's early slope OR its later plateau, never
        both at once. Handing the model the exact nonlinear quantity
        directly means training only has to learn ONE scalar weight
        on it (how much to trust this exact probability), instead of
        trying to reconstruct a saturating curve out of a straight
        line.
    is_accused_by_sole_seer / is_accused_by_contested_seer: some Sy
        (Seer) claimant has publicly reported this candidate as a
        wolf (main.py's seer_reports). Kept as TWO separate features
        (mirroring is_nominated_by_sole/is_nominated_by_contested) so
        training can weigh an UNCONTESTED Seer claimant's accusation
        more heavily than one from a claimant currently part of a
        contested pair (who might be the wolf trying to frame
        someone). A sole/uncontested claim alone is never a certainty
        on its own -- and, now that a Madman can also hold (and bluff
        from) a Sy claim, neither is a claimant CONFIRMED merely
        non-wolf; see is_accused_by_confirmed_real_seer below for that
        formerly-hard-rule case, now downgraded to a learned feature
        too.
    is_confirmed_contradicting_seer: this candidate's OWN "Sy" claim
        has been caught directly contradicting a report from a
        DIFFERENT Sy claimant already confirmed non-wolf (see
        _is_confirmed_contradicting_seer). This is a claim-fakeness
        proof, not a wolf-identity proof -- a Madman's bluffed "Sy"
        claim can be caught this way just as surely as a wolf's can --
        so it stays in the boosted-LR/decay tier (_CONFIRMED_TAIL_
        OFFSETS): "this specific claim is fake" remains 100% certain
        regardless of who's behind it, only "therefore always execute
        this fake claimant" stopped being automatically correct (a
        fake claimant might be a Madman, whose removal doesn't reduce
        the wolf headcount the way a fake wolf claimant's does -- see
        config.py's comment on "Kg" for the win-condition arithmetic).
    is_accused_by_confirmed_real_seer: some Sy claimant already found
        among confirmed_villagers (non-wolf by SOME independent means)
        has reported this candidate as a wolf. Before the Madman
        existed this was main.py's "詰み" (checkmate) hard rule --
        confirmed-non-wolf-and-claiming-Sy could only mean "the one
        genuine Seer", so the verdict was auto-executed on sight. Now
        a Madman can be confirmed non-wolf by the exact same means
        while bluffing "Sy", so this signal can also come from a
        Madman's FABRICATED report -- it is deliberately left OUT of
        the boosted tier above, and treated as an ordinary learned
        suspicion feature instead of a forced action.
    is_self_declared_non_village: this candidate has publicly declared
        hostility to the village (main.py's self_declared_non_village)
        -- either the Madman's genuine endgame "捨てCO" reveal
        (madman_ai.choose_endgame_reveal_action) or a genuine
        Villager's "村騙り" decoy bluff (villager_ai.choose_endgame_
        bluff_action) writing into the EXACT SAME list on purpose.
        Since either one can produce this fact, it can never be
        treated as proof of anything -- an ordinary, fully-learned
        suspicion feature only, deliberately excluded from
        _CONFIRMED_TAIL_OFFSETS's boosted tier (see that list's
        comment on this exact feature for why).
    is_this_voters_own_seer_accusation: SELF-REFERENTIAL, the Seer's
        counterpart to is_this_voters_own_nomination -- true only when
        THIS voter has personally, in their own public "Sy" report,
        accused `candidate` of being a wolf. Without this, the one
        genuine Seer's OWN vote has no way to tell "a candidate some
        Sy claimant accused" (is_accused_by_sole_seer /
        is_accused_by_contested_seer, shared by every village-side
        voter alike, including ones who can't be sure which claimant
        if any is honest) apart from "a candidate I MYSELF
        investigated and know for a fact is a wolf" -- collapsing the
        real Seer's own certainty down to the same ambiguous prior an
        ordinary Villager has to reason from. As _CONFIRMED_TAIL_
        OFFSETS's comment on this feature explains, this is not a
        probabilistic tell: main.py's run_vote sends every wolf voter
        through wolf_ai.choose_vote_target and every Madman voter
        through madman_ai.choose_vote_target regardless of what
        they've fake-claimed, so the only voter who can EVER reach
        this policy already holding a seer_reports entry naming
        themselves as claimant is the one real Seer, reporting their
        own genuine finding.
    is_accused_by_track_recorded_seer: some Sy claimant has accused
        `candidate` of being a wolf, AND that SAME claimant has, on
        some EARLIER report, also accused someone else who has SINCE
        been confirmed an actual wolf (via a trustworthy Medium
        report or guard-instruction verification -- see
        _extract_features's confirmed_wolf_ids). This is the missing
        piece for boards like (2,2,0)/(2,2,1) where NEITHER Sy
        claimant ever becomes individually confirmed non-wolf (the
        wolf-counting pigeonhole can't saturate with a Madman in
        play, and there's no Hunter claim to verify against either)
        -- without it, a Sy claimant who correctly calls out a gray
        wolf while their own identity remains permanently unresolved
        gets ZERO credit for having been right, and the village has
        no way to learn "keep trusting THIS specific claimant's next
        call more" the way it already can for a Medium's nomination
        (is_nominated_by_vindicated_contested_claimant) once that
        Medium is independently proven non-wolf. This is deliberately
        a TRACK-RECORD signal (was this claimant's past call correct?)
        rather than an IDENTITY signal (is this claimant proven non-
        wolf?) -- the two are independent: a claimant can rack up a
        correct call without their own identity ever being resolved,
        which is exactly the situation this exists to catch. Kept as
        an ordinary learned feature (not the boosted _CONFIRMED_TAIL_
        OFFSETS tier): a lucky guess, or a wolf/Madman deliberately
        sacrificing a teammate's cover to build fake credibility for a
        later lie, can't be ruled out, so this is a strong prior, not
        a certainty.
    is_last_wolf_pressure_by_claim (2026-10-05): the CLAIM-based twin of
        is_last_wolf_pressure. Counts wolves reported by a SOLE Medium
        claimant (state["claimed_wolf_ids"]) alongside the proven ones.
        A claim is not a proof -- a lone fake Medium looks exactly like
        a lone real one -- so this is a learned-trust feature, not a
        certainty (the old code only counted a real Medium's verdicts,
        which let the village tell a real lone Medium from a fake one).
    same_role_wolf_found (2026-10-05 user request): some OTHER player who
        claimed the SAME role as `candidate` has already been confirmed
        a wolf (a black Medium result on an executed claimant, a
        guard-verified fake Hunter, or a confirmed wolf by a real
        Seer). The human rule of thumb: once a roll of one role has
        already caught a wolf, the remaining claimants of that role are
        less likely to hide another one, so the roll can stop and the
        village can switch to gray. Only a feature -- how much it
        matters is for the policy to learn (a role can hold a wolf AND
        a Madman, or two wolves).
    off_line_vote_count (2026-10-05 user request): how many times THIS
        candidate has voted OUTSIDE that day's "village line" -- the set
        of all living claimants of the contested role (Sy first, then
        Ps, then Hn) that a roll would be working through. Voting for
        none of the roll candidates is the human "方針に従わない人は怪しい"
        signal. Only suggestive evidence (a real Seer can have a good
        reason to deviate; a wolf can follow the line to blend in), so
        it is a plain count the learned policy weighs for itself --
        see main.py's off_line_voters.
    is_confirmed_wolf_by_real_seer (2026-10-04 "ABC elimination"
        addition): `candidate` has been reported a wolf by a Sy
        claimant who is, with full IDENTITY certainty (state
        ["confirmed_real_claimants"] -- the wolf-counting pigeonhole,
        which accounts for the ENTIRE total_wolves + total_madmen
        budget before declaring anyone real), the genuine Seer. This
        is the user-requested "白占いの概念" (ABC-elimination) deduction:
        once two of three same-situation players are independently
        confirmed white/non-wolf and one claimant's identity is
        airtight, that claimant's own "black" verdict on the third is
        ground truth, not a guess -- reinstating (in a Madman-safe
        form) what the removed "Seer cascade" used to do unconditionally.
        Boosted to the same certainty tier as confirmed_real_hunter
        (see _CONFIRMED_TAIL_OFFSETS). Distinct from is_accused_by_
        confirmed_real_seer just above, which stays an ordinary,
        uncertain signal precisely because IT can come from a bluffing
        Madman's fabricated report -- this one structurally cannot.
    is_kg_claimant / is_sole_kg_claimant / contest_kg_wolf_probability:
        this candidate has publicly claimed "狂人CO" (main.py's
        endgame_claims, from the 3-4-players-remain identity-claim
        mechanic -- wolf_ai.ENDGAME_DECLARE_OPTIONS et al). Exactly ONE
        real Madman exists per config.ROLES, so this is EXACTLY the
        same "at most one genuine holder" structure a Ps/Hn/Sy claim
        already has, and gets the same shape of feature: is_sole_kg_
        claimant is trustworthy the moment nobody else has also claimed
        "狂人CO" (nothing yet PROVES it, the same way an uncontested
        Ps/Hn/Sy claim doesn't, but it's a real prior); contest_kg_
        wolf_probability is the exact same (count-1)/count quantity
        computed inline below for each CLAIMABLE_ROLES entry, here for
        the "狂人CO" claim pool instead -- 2+ simultaneous claimants
        proves AT LEAST one of them is lying (matching the user's own
        worked example: a "狂人CO" claim made once the real Madman is
        already independently accounted for elsewhere is provably
        fake, exactly mirroring a contested Sy claim). Not folded into
        the CLAIMABLE_ROLES loop itself because "狂人CO" isn't a
        Day-1-forced co_history claim the way Ps/Hn/Sy are
        (FORCED_DAY1_CLAIM_ROLES, is_impossible_late_forced_claim) --
        it's an optional, ANY-day endgame declaration from a completely
        separate mechanic, so it gets its own small, parallel block
        instead of pretending to fit the Day-1 claim machinery's
        assumptions.
    is_lone_survivor_of_contest (per claimable role, in the role loop
        below): this candidate's OWN claim used to be contested
        (2+ EVER claimed it) but every OTHER claimant of that same
        role has since died, leaving this candidate the sole one
        still alive. Distinct from is_sole (which means nobody else
        EVER claimed it -- a much stronger, from-day-1 situation):
        a rival dying tells you nothing by itself about who was
        genuine (an execution or night-kill doesn't reveal a role,
        only sometimes a later Medium check does) -- so this is NOT
        folded into is_sole's "trust immediately" bucket, which would
        overstate the certainty. It's kept as its own separately-
        weighted, ordinary learned feature so training can decide how
        much (if anything) attrition-without-proof is worth trusting,
        rather than the previous behavior of silently continuing to
        score this candidate as if the contest were still fully live
        (is_accused_by_contested_seer / nominated_by_contested /
        is_last_wolf_pressure's is_unresolved check all already
        switched to ALIVE claim counts for the same reason -- see
        _extract_features).
    """

    features = [1.0]

    for role in CLAIMABLE_ROLES:
        is_this = 1.0 if own_co == role else 0.0
        count = claim_counts.get(role, 0)
        is_sole = 1.0 if (is_this and count == 1) else 0.0
        is_fake = 1.0 if (is_this and own_co == provably_fake_role) else 0.0
        contest_wolf_prob = ((count - 1) / count) if (is_this and count >= 1) else 0.0
        alive_count_for_role = (
            alive_claim_counts.get(role, count) if alive_claim_counts is not None else count
        )
        is_lone_survivor = 1.0 if (is_this and count >= 2 and alive_count_for_role <= 1) else 0.0
        features.extend([is_this, is_this * count, is_sole, is_fake, contest_wolf_prob, is_lone_survivor])

    features.extend([
        1.0 if is_villager_claim else 0.0,
        1.0 if is_confirmed_villager else 0.0,
        1.0 if is_confirmed_fake_hunter else 0.0,
        1.0 if is_confirmed_real_hunter else 0.0,
        alive_count / 7.0,
        guard_count / 2.0,
        1.0 if is_falsely_vouched else 0.0,
        1.0 if is_pending_verification else 0.0,
        1.0 if is_nominated_by_sole else 0.0,
        1.0 if is_nominated_by_contested else 0.0,
        1.0 if is_impossible_late_forced_claim else 0.0,
        1.0 if is_confirmed_contradicting_medium else 0.0,
        1.0 if is_last_wolf_pressure else 0.0,
        1.0 if is_vouched_for_executed else 0.0,
        suspicious_non_vote_count / 2.0,
        1.0 if is_nominated_by_vindicated_contested_claimant else 0.0,
        1.0 if is_this_voters_own_nomination else 0.0,
        defied_sole_nomination_count / 2.0,
        voted_for_protected_count / 2.0,
        1.0 if is_accused_by_sole_seer else 0.0,
        1.0 if is_accused_by_contested_seer else 0.0,
        1.0 if is_confirmed_contradicting_seer else 0.0,
        1.0 if is_accused_by_confirmed_real_seer else 0.0,
        1.0 if is_self_declared_non_village else 0.0,
        1.0 if is_this_voters_own_seer_accusation else 0.0,
        1.0 if is_kg_claimant else 0.0,
        1.0 if is_sole_kg_claimant else 0.0,
        contest_kg_wolf_probability,
        1.0 if is_accused_by_track_recorded_seer else 0.0,
        1.0 if is_confirmed_wolf_by_real_seer else 0.0,
        off_line_vote_count / 2.0,
        1.0 if same_role_wolf_found else 0.0,
        1.0 if is_last_wolf_pressure_by_claim else 0.0,
    ])
    features.extend(extra if extra is not None else [0.0] * NUM_EXTRA_TAIL_FEATURES)

    return features


FORCED_DAY1_CLAIM_ROLES = ("Ps", "Sy")


def _is_impossible_late_forced_claim(state, candidate):
    """
    True if `candidate`'s "Ps" or "Sy" claim is structurally
    impossible for the real Medium/Seer, BOTH of whom are forced to
    reveal on Day 1 and never get another chance (main.py's
    medium_day_co and seer_day_co). Two separate ways this can be
    certain, neither needing any comparison to another claimant:

      1. Claimed on any day OTHER than Day 1.

      2. Claimed via the last-second emergency-evasion path (main.py's
         progression_ai's wolf check, tracked in emergency_wolf_claims)
         -- even if that happened to still be Day 1. A ROUTINE same-day
         fake claim (via wolf_day_co) is deliberately treated as
         genuinely ambiguous "simultaneous morning announcement"
         timing, since the code processing one player's claim before
         another's within that same phase is just an implementation
         detail, not something a real observer could reliably order.
         But the emergency path only runs AFTER that day's CO phase
         AND that day's nomination have already happened -- so a
         "Ps"/"Sy" claim from it is unambiguously, observably LATER
         than the real Medium's/Seer's mandatory reveal (which always
         happens first, before nominations even exist). That ordering
         isn't a simulation artifact; it's a genuine in-fiction fact
         anyone watching the game would see for themselves ("you only
         spoke up after being accused").

    Before the Seer was also forced to reveal Day 1, this only ever
    applied to "Ps" (hence its old name, _is_impossible_late_medium_
    claim); a late "Sy" claim used to be genuinely ambiguous, since a
    real Seer revealing on Day 3 was a normal, allowed strategic
    choice. Now that both roles share the exact same forced-reveal
    rule, the same structural proof applies to both -- see
    FORCED_DAY1_CLAIM_ROLES.
    """
    own_co = state["co_history"][candidate]
    if own_co not in FORCED_DAY1_CLAIM_ROLES:
        return False

    claim_day = state.get("claim_day")
    if (
        claim_day is not None
        and claim_day[candidate] is not None
        and claim_day[candidate] != 1
    ):
        return True

    if candidate in state.get("emergency_wolf_claims", []):
        return True

    return False


def _is_confirmed_contradicting_medium(state, candidate):
    """
    True if `candidate` (a fake Medium claimant) has EVER reported
    a verdict on some player that disagrees with the real Medium's
    verdict on that SAME player -- an absolute contradiction.
    """
    # 2026-10-06 (L2): built from the symmetric public claims list, not
    # a key that existed only for fake claimants.
    fake_reports = state.get("medium_claims", [])
    real_reports = state.get("medium_reports", [])
    real_by_player = {r["player"]: r["is_wolf"] for r in real_reports}

    for report in fake_reports:
        if report["claimant"] != candidate:
            continue
        real_verdict = real_by_player.get(report["player"])
        if real_verdict is not None and report["is_wolf"] != real_verdict:
            return True

    return False


def _is_confirmed_contradicting_seer(state, candidate):
    """
    True if `candidate` (a Sy claimant) has reported a verdict on some
    player that directly disagrees with a report from a DIFFERENT Sy
    claimant already known to be the genuine Seer.

    "Already known to be genuine" here means simply: found among
    state["confirmed_villagers"] while claiming "Sy" (see main.py's
    build_public_state) -- since only wolves ever fake a role claim
    in this game (main.py's villager_day_co, the option for a genuine
    Villager to bluff, is disabled), a non-wolf Sy claimant can only
    be the real Seer, with total certainty, regardless of HOW they
    came to be confirmed non-wolf (a night-attack survival, guard-
    instruction verification, or the wolf-counting pigeonhole
    deduction). Once that's established, their reports are ground
    truth, and anyone else's report that flatly disagrees about the
    SAME target must be a lie -- the same absolute-contradiction
    standard _is_confirmed_contradicting_medium already uses.

    2026-10-04: a same-day attempt to tighten this to state[
    "confirmed_real_claimants"] (closing a theoretical Madman-trap:
    a Madman confirmed non-wolf by merely surviving/dying to an attack
    is NOT thereby proven to be the genuine Seer, so trusting their
    fabricated report here could in principle convict the real Seer's
    true, contradicting report instead) was REVERTED after measuring a
    severe regression in live training -- see _is_confirmed_fake_by_
    cleared_rival's matching note for the measurement and reasoning:
    the common case this rule exists for is far more frequent than the
    narrow Madman-coincidence it's theoretically vulnerable to. The
    narrow case is instead covered, WITHOUT this tradeoff, by the new
    is_confirmed_wolf_by_real_seer feature (fed only from the airtight
    confirmed_real_claimants/confirmed_wolves data computed alongside
    this).
    """
    co_history = state["co_history"]
    if co_history[candidate] != "Sy":
        return False

    confirmed_villagers = state.get("confirmed_villagers", [])
    confirmed_real_seers = [
        i for i in confirmed_villagers
        if i != candidate and co_history[i] == "Sy"
    ]
    if not confirmed_real_seers:
        return False

    reports = state.get("seer_reports", [])
    real_verdict_by_target = {}
    for r in reports:
        if r["claimant"] in confirmed_real_seers:
            real_verdict_by_target[r["player"]] = r["is_wolf"]

    for r in reports:
        if r["claimant"] != candidate:
            continue
        real_verdict = real_verdict_by_target.get(r["player"])
        if real_verdict is not None and r["is_wolf"] != real_verdict:
            return True

    return False


def _is_hard_proven_fake_claim(state, candidate):
    """
    True if `candidate`'s OWN claim has been proven fake by an
    unconditional, 100%-certain rule: an impossible late Medium OR
    Seer claim (both roles are now forced to reveal Day 1 -- see
    _is_impossible_late_forced_claim), a Medium claimant directly
    contradicting the real Medium's own report, or a Seer claimant
    directly contradicting an already-confirmed real Seer's own
    report.

    This is a CLAIM-FAKENESS proof, not a WOLF-IDENTITY proof, and the
    two stopped being interchangeable once the Madman was added: a
    Madman can fake-claim Ps/Hn/Sy exactly like a wolf can, so a
    "this claim is definitely fake" verdict no longer tells you which
    side is behind it. choose_execution_target therefore no longer
    auto-executes on this alone -- see its docstring for why
    (removing a Madman doesn't shrink the wolf headcount the way
    removing a wolf does, so it isn't even unconditionally GOOD for
    the village the way it always used to be). This function still
    feeds the (boosted-LR/decay) `is_fake` feature, and is still used
    below to keep an already-proven-fake claim from being able to
    "contest" another claimant of the same role for protection
    purposes -- both of those uses only need "this claim is fake",
    which remains exactly as certain as before.

    `co_history` never clears a dead player's claim (main.py keeps it
    for good reason -- see _wolf_counting_deduction), so the
    protection-floor exclusion below has to hold regardless of
    whether `candidate` is still alive.
    """
    return (
        _is_impossible_late_forced_claim(state, candidate)
        or _is_confirmed_contradicting_medium(state, candidate)
        or _is_confirmed_contradicting_seer(state, candidate)
    )


def _is_unresolved_contested_claimant(state, candidate, own_co, alive_claim_counts):
    """
    True if `candidate` is part of a role-claim that is CURRENTLY
    contested (2+ claimants STILL ALIVE today) and this specific
    candidate hasn't already been individually confirmed real or fake
    some other way. A contested claim is GUARANTEED to contain at
    least one wolf (only wolves fake claims here), so this marks "one
    of the people a wolf could still be hiding among."

    Takes ALIVE claim counts (not the raw ever-claimed co_history
    tally) -- once one of two claimants of a role has died, there is
    no longer a live choice to make "between" them for execution
    purposes, so a lone survivor should stop being treated as an
    active, resolvable-today contest by is_last_wolf_pressure. This
    is separate from (and doesn't imply) that survivor being TRUSTED
    -- see is_lone_survivor_of_contest in _build_feature_vector's role
    loop for the learned-trust side of that same situation.
    """
    if own_co not in CLAIMABLE_ROLES:
        return False
    if alive_claim_counts.get(own_co, 0) < 2:
        return False
    if own_co == "Hn":
        if candidate in state.get("confirmed_fake_hunters", []):
            return False
        if candidate == state.get("confirmed_real_hunter"):
            return False
    if own_co == "Ps":
        if _is_confirmed_contradicting_medium(state, candidate):
            return False
    return True


def _extra_features(state, candidate, own_co, alive_count, confirmed_wolf_ids):
    """
    2026-10-06 feature additions (all built ONLY from public state, or
    from the voter's own private seer results when the voter is the
    real Seer -- state["voter_private_whites"/"voter_private_blacks"]):

    [0] voted_for_confirmed_white_count/2: how often `candidate` voted
        (any past day) for someone now publicly cleared (night victim,
        trusted-white, medium-white). Human "who voted a white" logic.
    [1] voted_for_confirmed_wolf_count/2: the reverse (a good sign).
    [2..4] is_whited_by_sole_seer / _contested_seer / _confirmed_real_seer
        (white circle). [5] is_unreported_gray: no claim and nobody has
        a seer result on them.
    [6] seer_medium_conflict_count/2: candidate's own seer/medium
        verdicts that disagree with the verdict on the same player from
        the SOLE (or proven) claimant of the other role.
    [7] false_result_count/2: PUBLICLY PROVABLE lies -- a "black" on
        someone proven non-wolf (night victim etc.), or a "white" on a
        proven wolf. Certain, so boosted in _CONFIRMED_TAIL_OFFSETS.
    [8] is_my_same_role_rival: the voter claimed the same role as
        `candidate` (to a genuine holder that rival is certainly fake).
    [9]/[10] is_own_white / is_own_black: the real Seer voter's own
        private results on `candidate`.
    [11..15] rope/late-game interactions with the candidate's status:
        late = alive<=5, final = alive<=3.
    """
    co_history = state["co_history"]
    seer_reports = state.get("seer_reports", [])
    medium_claims = state.get("medium_claims", [])
    medium_reports = state.get("medium_reports", [])
    confirmed_villagers = set(state.get("confirmed_villagers", []))
    real_claimants = set(state.get("confirmed_real_claimants", []))
    voter = state.get("voter")

    white_set = set(confirmed_villagers) | {r["player"] for r in medium_reports if not r["is_wolf"]}
    # (confirmed fake hunters are deliberately NOT here: a fake Hunter may
    # be the Madman, and a real Seer's white on him is not a lie.)
    wolf_set = set(state.get("confirmed_wolves", [])) | {r["player"] for r in medium_reports if r["is_wolf"]}
    voted_white = voted_wolf = 0
    for _day, votes in (state.get("vote_history") or {}).items():
        t = votes.get(candidate)
        if t is None or t == candidate:
            continue
        if t in white_set:
            voted_white += 1
        elif t in wolf_set:
            voted_wolf += 1

    sy_count = sum(1 for co in co_history if co == "Sy")
    ps_count = sum(1 for co in co_history if co == "Ps")
    whited_sole = whited_contested = whited_real = False
    seer_report_names_candidate = False
    for r in seer_reports:
        if r["player"] != candidate:
            continue
        seer_report_names_candidate = True
        if r["is_wolf"]:
            continue
        if r["claimant"] in real_claimants:
            whited_real = True
        if sy_count <= 1:
            whited_sole = True
        else:
            whited_contested = True
    is_gray = co_history[candidate] is None
    is_unreported_gray = is_gray and not seer_report_names_candidate and candidate not in white_set

    def _trusted_verdicts_seer():
        out = {}
        for r in seer_reports:
            if sy_count <= 1 or r["claimant"] in real_claimants:
                out[(r["claimant"], r["player"])] = r["is_wolf"]
        return out

    def _trusted_verdicts_medium():
        out = {(None, r["player"]): r["is_wolf"] for r in medium_reports}
        for r in medium_claims:
            if ps_count <= 1 or r["claimant"] in real_claimants:
                out[(r["claimant"], r["player"])] = r["is_wolf"]
        return out

    conflicts = 0
    if own_co == "Sy":
        other = {}
        for (c, p), v in _trusted_verdicts_medium().items():
            other[p] = v
        for r in seer_reports:
            if r["claimant"] == candidate and r["player"] in other and other[r["player"]] != r["is_wolf"]:
                conflicts += 1
    elif own_co == "Ps":
        other = {}
        for (c, p), v in _trusted_verdicts_seer().items():
            other[p] = v
        for r in medium_claims:
            if r["claimant"] == candidate and r["player"] in other and other[r["player"]] != r["is_wolf"]:
                conflicts += 1

    false_results = 0
    if own_co == "Sy":
        for r in seer_reports:
            if r["claimant"] != candidate:
                continue
            if r["is_wolf"] and r["player"] in confirmed_villagers:
                false_results += 1
            elif (not r["is_wolf"]) and r["player"] in wolf_set:
                false_results += 1

    my_co = co_history[voter] if voter is not None else None
    is_my_rival = (
        voter is not None and candidate != voter
        and my_co in CLAIMABLE_ROLES and my_co == own_co
    )
    is_own_white = candidate in state.get("voter_private_whites", [])
    is_own_black = candidate in state.get("voter_private_blacks", [])

    late = 1.0 if alive_count <= 5 else 0.0
    final = 1.0 if alive_count <= 3 else 0.0
    is_claimant = 1.0 if own_co in CLAIMABLE_ROLES else 0.0
    is_sole = 1.0 if (own_co in CLAIMABLE_ROLES and
                      sum(1 for co in co_history if co == own_co) == 1) else 0.0
    gray = 1.0 if is_gray else 0.0

    # ---- 2026-10-06 (second batch) ---------------------------------
    def _false_count(claimant):
        n = 0
        for r in seer_reports:
            if r["claimant"] != claimant:
                continue
            if r["is_wolf"] and r["player"] in confirmed_villagers:
                n += 1
            elif (not r["is_wolf"]) and r["player"] in wolf_set:
                n += 1
        return n

    sy_claimants = [i for i, co in enumerate(co_history) if co == "Sy"]
    proven_fake_sy = {
        s for s in sy_claimants
        if _is_hard_proven_fake_claim(state, s) or _false_count(s) > 0
    }
    on_cand = [r for r in seer_reports if r["player"] == candidate]
    reporters = {r["claimant"] for r in on_cand}
    verdicts = {r["is_wolf"] for r in on_cand}
    all_agree_white = len(reporters) >= 2 and verdicts == {False}
    all_agree_black = len(reporters) >= 2 and verdicts == {True}
    is_split = len(reporters) >= 2 and len(verdicts) == 2
    black_from_fake = any(r["is_wolf"] and r["claimant"] in proven_fake_sy for r in on_cand)
    white_from_fake = any((not r["is_wolf"]) and r["claimant"] in proven_fake_sy for r in on_cand)

    rival_ps_black = rival_ps_white = False
    if own_co == "Ps":
        for r in medium_claims:
            if r["claimant"] == candidate and co_history[r["player"]] == "Ps" and r["player"] != candidate:
                if r["is_wolf"]:
                    rival_ps_black = True
                else:
                    rival_ps_white = True

    board_alive = [i for i, b in enumerate(state.get("board", [])) if b != "--"]
    ropes_left = max(0, (alive_count - 1) // 2)
    wolves_left_hat = max(0, state.get("total_wolves", 2) - state.get("confirmed_dead_wolves", 0))
    slack = ropes_left - wolves_left_hat
    tight = 1.0 if slack <= 0 else 0.0
    loose = 1.0 if slack >= 1 else 0.0
    is_confirmed_v = 1.0 if candidate in confirmed_villagers else 0.0
    no_hn = 1.0 if not any(co == "Hn" for co in co_history) else 0.0
    gray_unreported = sum(
        1 for i in board_alive
        if co_history[i] is None and i not in white_set
        and not any(r["player"] == i for r in seer_reports)
    )
    gray_ratio = min(gray_unreported / max(1, ropes_left), 3.0) / 3.0

    truth = state.get("voter_private_truth", {})
    private_contra = 0
    if truth:
        for r in seer_reports + medium_claims:
            if r["claimant"] == candidate and r["player"] in truth and truth[r["player"]] != r["is_wolf"]:
                private_contra += 1

    nominator_inc = state.get("nominator_inconsistency", {}).get(candidate, 0)
    not_voting_black = state.get("not_voting_own_black", {}).get(candidate, 0)

    claimed_white = {r["player"] for r in medium_claims if (not r["is_wolf"]) and ps_count <= 1}
    voted_claimed_white = 0
    for _day, votes in (state.get("vote_history") or {}).items():
        t = votes.get(candidate)
        if t is not None and t != candidate and t in claimed_white:
            voted_claimed_white += 1

    return [
        voted_white / 2.0,
        voted_wolf / 2.0,
        1.0 if whited_sole else 0.0,
        1.0 if whited_contested else 0.0,
        1.0 if whited_real else 0.0,
        1.0 if is_unreported_gray else 0.0,
        conflicts / 2.0,
        false_results / 2.0,
        1.0 if is_my_rival else 0.0,
        1.0 if is_own_white else 0.0,
        1.0 if is_own_black else 0.0,
        late * is_claimant,
        late * is_sole,
        late * gray,
        final * is_claimant,
        final * gray,
        # ---- second batch (indices 16..33) ----
        1.0 if all_agree_white else 0.0,
        1.0 if all_agree_black else 0.0,
        1.0 if is_split else 0.0,
        1.0 if black_from_fake else 0.0,
        1.0 if white_from_fake else 0.0,
        1.0 if rival_ps_black else 0.0,
        1.0 if rival_ps_white else 0.0,
        tight * is_sole,
        tight * is_confirmed_v,
        tight * gray,
        tight * is_claimant,
        loose * is_claimant,
        no_hn * gray,
        gray_ratio * (1.0 if is_unreported_gray else 0.0),
        private_contra / 2.0,
        nominator_inc / 2.0,
        not_voting_black / 2.0,
        voted_claimed_white / 2.0,
    ]



def _extract_features(state, candidate):

    co_history = state["co_history"]
    alive_count = state.get("alive_count", sum(1 for s in state["board"] if s != "--"))
    guard_count = state.get("guard_success_count", 0)

    board = state["board"]
    own_co = co_history[candidate]
    claim_counts = {role: sum(1 for co in co_history if co == role) for role in CLAIMABLE_ROLES}
    # ALIVE-only counterpart to claim_counts above. Used wherever the
    # question is "is this role's claim STILL an open, live contest
    # today" (nominated_by_sole/contested, is_accused_by_sole/
    # contested_seer, _is_unresolved_contested_claimant) rather than
    # "how many people have EVER claimed it" (claim_counts itself,
    # kept as-is for is_sole/contest_wolf_probability's pigeonhole
    # math, which doesn't change just because someone died with no
    # further information revealed). Before this, a role's claim
    # count never dropped even after one claimant died, so a lone
    # surviving claimant kept being scored exactly as ambiguous as a
    # still-fully-live 2-way race forever -- see is_lone_survivor_of_
    # contest's comment on _build_feature_vector for the learned-trust
    # half of this same fix.
    alive_claim_counts = {
        role: sum(1 for i, co in enumerate(co_history) if co == role and board[i] != "--")
        for role in CLAIMABLE_ROLES
    }
    # 2026-10-06 (round 4): EVER-claimed count, same as main.py's
    # run_vote -- a dead rival Medium claimant still makes the claim
    # contested.
    medium_claimant_count = claim_counts.get("Ps", 0)

    is_provably_fake = (
        _is_provably_fake(state, candidate, own_co)
        or _is_confirmed_fake_by_cleared_rival(state, candidate, own_co)
        or _is_self_contradicting_seer_claim(state, candidate, own_co)
    )

    claim_day = state.get("claim_day")
    is_impossible_late_forced_claim = _is_impossible_late_forced_claim(state, candidate)

    is_confirmed_contradicting_medium = (
        own_co == "Ps" and _is_confirmed_contradicting_medium(state, candidate)
    )

    total_wolves = state.get("total_wolves", 2)
    # NOTE: despite the key name (kept for main.py compatibility),
    # this counts every wolf CONFIRMED so far by any means -- via a
    # Medium report on an executed player, OR via guard-instruction
    # verification (confirmed_fake_hunters) -- regardless of
    # whether that wolf happens to still be alive. A wolf identified
    # while still alive is just as useful for this deduction as one
    # already dead.
    confirmed_identified_wolves = state.get("confirmed_dead_wolves", 0)
    is_last_wolf_pressure = (
        confirmed_identified_wolves == total_wolves - 1
        and _is_unresolved_contested_claimant(state, candidate, own_co, alive_claim_counts)
    )

    execution_history = state.get("execution_history", [])
    guard_vouches = state.get("guard_vouches", {})
    is_vouched_for_executed = any(
        t in execution_history for t in guard_vouches.get(candidate, [])
    )

    seer_reports = state.get("seer_reports", [])
    # 2026-10-06 (re-audit H1): "sole" must mean nobody ELSE ever claimed
    # Sy -- not "only one is alive right now" (that made a dead rival's
    # lying black count as a sole seer's high-trust black once the
    # rival was executed and the village still did not know who was real).
    seer_claim_count = claim_counts.get("Sy", 0)

    # confirmed_wolf_ids: identities (not just the count main.py's
    # "confirmed_dead_wolves" exposes) of every player confirmed a
    # wolf so far by either means -- mirrors main.py's build_public_
    # state formula for that count exactly, just keeping the actual
    # player indices instead of collapsing them to len(set(...)).
    # Used below to check a Sy claimant's OWN track record.
    confirmed_wolf_ids = (
        set(r["player"] for r in state.get("medium_reports", []) if r["is_wolf"])
        | set(state.get("confirmed_fake_hunters", []))
        | set(state.get("confirmed_wolves", []))
    )

    # is_confirmed_wolf_by_real_seer: `candidate` appears in
    # state["confirmed_wolves"] -- main.py's build_public_state only
    # ever populates this from a Sy claimant found in confirmed_real_
    # claimants (the airtight, identity-proven pigeonhole case), never
    # from mere non-wolf status, so this is a 100%-certain fact, not a
    # suspicion -- see _CONFIRMED_TAIL_OFFSETS's matching NOTE.
    is_confirmed_wolf_by_real_seer = candidate in state.get("confirmed_wolves", [])

    is_accused_by_sole_seer = False
    is_accused_by_contested_seer = False
    is_accused_by_track_recorded_seer = False
    for r in seer_reports:
        if r["player"] != candidate or not r["is_wolf"]:
            continue
        if seer_claim_count <= 1:
            is_accused_by_sole_seer = True
        else:
            is_accused_by_contested_seer = True
        claimant = r["claimant"]
        has_earlier_vindicated_call = any(
            r2["claimant"] == claimant and r2["is_wolf"] and r2["player"] in confirmed_wolf_ids
            for r2 in seer_reports
        )
        if has_earlier_vindicated_call:
            is_accused_by_track_recorded_seer = True

    # is_this_voters_own_seer_accusation: see _build_feature_vector's
    # docstring and _CONFIRMED_TAIL_OFFSETS's comment for why checking
    # "did THIS voter personally file a seer_reports entry naming
    # `candidate` a wolf" is safe to trust with total certainty here --
    # no wolf or Madman voter, however they've fake-claimed, ever
    # reaches this feature-extraction path for their OWN vote.
    voter = state.get("voter")
    is_this_voters_own_seer_accusation = any(
        r["claimant"] == voter and r["player"] == candidate and r["is_wolf"]
        for r in seer_reports
    )

    nominations = state.get("day1_nomination") or state.get("nominations", {})
    nominated_by_sole = False
    nominated_by_contested = False
    nominated_by_vindicated_contested = False
    confirmed_villagers = state.get("confirmed_villagers", [])
    voter = state.get("voter")
    is_own_nomination = False
    for nominator, target in nominations.items():
        if target != candidate:
            continue
        if medium_claimant_count <= 1:
            nominated_by_sole = True
        else:
            nominated_by_contested = True
            if nominator in confirmed_villagers:
                nominated_by_vindicated_contested = True
        if voter is not None and nominator == voter:
            is_own_nomination = True

    is_confirmed_contradicting_seer = (
        own_co == "Sy" and _is_confirmed_contradicting_seer(state, candidate)
    )

    # is_accused_by_confirmed_real_seer: a Sy claimant already found
    # among confirmed_villagers (non-wolf by SOME independent means --
    # a night-attack survival, guard-instruction verification, or the
    # wolf-counting pigeonhole; see _is_confirmed_contradicting_seer's
    # docstring for the same reasoning) has reported `candidate` as a
    # wolf. Before the Madman existed this WAS the "詰み" checkmate --
    # main.py's build_public_state used to hand it straight through as
    # state["seer_confirmed_wolf_accusations"] and choose_execution_
    # target auto-executed on sight, because a confirmed-non-wolf Sy
    # claimant could only be the one genuine Seer, full stop. Now a
    # Madman can ALSO be confirmed non-wolf (that confirmation never
    # checks literal role, only "not a wolf") while bluffing "Sy" --
    # so this same signal can now come from a Madman's FABRICATED
    # report instead of the real Seer's. It's computed locally here
    # (rather than trusted from state, or used to mutate
    # confirmed_villagers the way main.py's old cascade did) and fed
    # in as an ordinary learned feature, deliberately EXCLUDED from
    # the boosted-LR/decay tier _CONFIRMED_TAIL_OFFSETS gives
    # is_confirmed_contradicting_seer -- see that list's comment.
    is_accused_by_confirmed_real_seer = False
    if candidate not in state.get("confirmed_villagers", []):
        confirmed_seer_claimants = [
            i for i in confirmed_villagers
            if i != candidate and co_history[i] == "Sy"
        ]
        for r in seer_reports:
            if r["claimant"] in confirmed_seer_claimants and r["player"] == candidate and r["is_wolf"]:
                is_accused_by_confirmed_real_seer = True
                break

    endgame_claims = state.get("endgame_claims", {})
    kg_claim_count = sum(1 for v in endgame_claims.values() if v == "madman")
    is_kg_claimant = endgame_claims.get(candidate) == "madman"
    is_sole_kg_claimant = is_kg_claimant and kg_claim_count == 1
    contest_kg_wolf_probability = (
        (kg_claim_count - 1) / kg_claim_count if is_kg_claimant and kg_claim_count >= 1 else 0.0
    )

    return _build_feature_vector(
        own_co=own_co,
        claim_counts=claim_counts,
        alive_claim_counts=alive_claim_counts,
        is_villager_claim=(own_co == "Vi"),
        is_confirmed_villager=candidate in state.get("confirmed_villagers", []),
        is_confirmed_fake_hunter=candidate in state.get("confirmed_fake_hunters", []),
        is_confirmed_real_hunter=candidate == state.get("confirmed_real_hunter"),
        alive_count=alive_count,
        guard_count=guard_count,
        provably_fake_role=own_co if is_provably_fake else None,
        is_falsely_vouched=candidate in state.get("falsely_vouched_by_fake", []),
        is_pending_verification=candidate in state.get("guard_instructions", {}),
        is_last_wolf_pressure=is_last_wolf_pressure,
        is_vouched_for_executed=is_vouched_for_executed,
        is_impossible_late_forced_claim=is_impossible_late_forced_claim,
        is_confirmed_contradicting_medium=is_confirmed_contradicting_medium,
        is_nominated_by_sole=nominated_by_sole,
        is_nominated_by_contested=nominated_by_contested,
        suspicious_non_vote_count=state.get("suspicious_non_voters", {}).get(candidate, 0),
        is_nominated_by_vindicated_contested_claimant=nominated_by_vindicated_contested,
        is_this_voters_own_nomination=is_own_nomination,
        defied_sole_nomination_count=state.get("defied_sole_nomination_count", {}).get(candidate, 0),
        voted_for_protected_count=state.get("voted_for_protected_count", {}).get(candidate, 0),
        is_accused_by_sole_seer=is_accused_by_sole_seer,
        is_accused_by_contested_seer=is_accused_by_contested_seer,
        is_confirmed_contradicting_seer=is_confirmed_contradicting_seer,
        is_accused_by_confirmed_real_seer=is_accused_by_confirmed_real_seer,
        is_self_declared_non_village=(
            candidate in state.get("self_declared_non_village", [])
        ),
        is_this_voters_own_seer_accusation=is_this_voters_own_seer_accusation,
        is_kg_claimant=is_kg_claimant,
        is_sole_kg_claimant=is_sole_kg_claimant,
        contest_kg_wolf_probability=contest_kg_wolf_probability,
        is_accused_by_track_recorded_seer=is_accused_by_track_recorded_seer,
        is_confirmed_wolf_by_real_seer=is_confirmed_wolf_by_real_seer,
        off_line_vote_count=state.get("off_line_voters", {}).get(candidate, 0),
        same_role_wolf_found=(
            own_co in CLAIMABLE_ROLES
            and any(co_history[w] == own_co
                    for w in (set(confirmed_wolf_ids) | set(state.get("claimed_wolf_ids", [])))
                    if w != candidate)
        ),
        is_last_wolf_pressure_by_claim=(
            not is_last_wolf_pressure
            and len(set(confirmed_wolf_ids) | set(state.get("claimed_wolf_ids", [])))
                == total_wolves - 1
            and _is_unresolved_contested_claimant(state, candidate, own_co, alive_claim_counts)
        ),
        extra=_extra_features(state, candidate, own_co, alive_count, confirmed_wolf_ids),
    )


def lookup(alive_count, own_co, medium_count=0, hunter_count=0, guard_count=0,

           is_confirmed_villager=False, is_confirmed_fake_hunter=False,
           is_confirmed_real_hunter=False, is_provably_fake_claim=False,
           is_falsely_vouched=False, is_pending_verification=False,
           is_nominated_by_sole=False, is_nominated_by_contested=False,
           is_impossible_late_forced_claim=False,
           is_confirmed_contradicting_medium=False,
           is_last_wolf_pressure=False,
           is_vouched_for_executed=False):
    """
    Convenience accessor for reporting: score for a specific,
    human-described situation. Kept in terms of THIS composition's
    specific roles (Medium/Hunter) for readability, but internally
    goes through the same generalized feature construction used
    during actual play.
    """
    claim_counts = {"Ps": medium_count, "Hn": hunter_count}
    provably_fake_role = own_co if is_provably_fake_claim else None

    features = _build_feature_vector(
        own_co=own_co,
        claim_counts=claim_counts,
        is_villager_claim=(own_co == "Vi"),
        is_confirmed_villager=is_confirmed_villager,
        is_confirmed_fake_hunter=is_confirmed_fake_hunter,
        is_confirmed_real_hunter=is_confirmed_real_hunter,
        alive_count=alive_count,
        guard_count=guard_count,
        provably_fake_role=provably_fake_role,
        is_falsely_vouched=is_falsely_vouched,
        is_pending_verification=is_pending_verification,
        is_nominated_by_sole=is_nominated_by_sole,
        is_nominated_by_contested=is_nominated_by_contested,
        is_impossible_late_forced_claim=is_impossible_late_forced_claim,
        is_confirmed_contradicting_medium=is_confirmed_contradicting_medium,
        is_last_wolf_pressure=is_last_wolf_pressure,
        is_vouched_for_executed=is_vouched_for_executed,
    )

    return execution_policy.score(features)


# ============================================================
# Guard-instruction assignment strategy (when 2 Hunter claimants
# need to be given DIFFERENT watch targets -- see main.py's
# assign_guard_instructions).
#
# Each of the 2 watch slots is independently assigned one of 3
# TARGET CATEGORIES: a random gray player, the Medium, or the
# OTHER claimant -- giving 3x3=9 combinations total. This is a
# flat, unconditional 9-option choice (no situational features
# like alive_count/day are mixed in), NOT a lookup table: it's
# the same shape as the previous 2-option version, just with a
# richer option set, so it doesn't reintroduce the fragmentation
# problem tables had earlier in this project.
# ============================================================

_TARGET_TYPES = ["gray", "medium", "hunter", "seer"]

# ============================================================
# 1-claimant (sole/uncontested Hunter claim) version (2026-10-04 user
# request). Before this, main.py's assign_guard_instructions refused
# to assign ANY guard instruction at all unless 2+ claimants were
# unresolved -- so a single, uncontested "Hn" claimant got ZERO of the
# verification this whole mechanism provides (announce a guard target,
# then prove fake/genuine from whether that target survives the
# night). That's exactly the gap behind this project's single largest
# underperforming board bucket (a lone fake Hunter claim with the real
# Hunter silently hiding, ~51% village win rate on over 40,000 games
# at the time this was diagnosed -- see this session's game_db-backed
# win_rate_by_detail() breakdown).
#
# Only 3 target categories make sense for a SOLE claimant -- "hunter"
# (main.py's cyclic "one of the OTHER claimants" rule) has no OTHER
# claimant to point to when there's only one, so it's deliberately
# left out of this option set rather than degenerating into
# self-targeting.
#
# Separate, freshly-initialized policy (not a reuse of the 2-slot
# policy's weights) -- same precedent as guard_instruction_policy_3
# above: the 1-claimant and 2-claimant cases are different enough
# situations (no second claimant's behavior to read at all) that
# sharing weights would mix two different decisions into one.
# ============================================================

_TARGET_TYPES_SOLO = ["gray", "medium", "seer"]

GUARD_INSTRUCTION_OPTIONS_1 = list(_TARGET_TYPES_SOLO)

guard_instruction_policy_1 = policy_utils.LinearPolicy(len(GUARD_INSTRUCTION_OPTIONS_1))
guard_instruction_split_1 = board_split.BoardSplitManager(guard_instruction_policy_1)


def choose_guard_instruction_strategy_1(state):
    """
    Decide the single watch slot's target category when exactly ONE
    Hunter claimant is unresolved: one of "gray"/"medium"/"seer"
    (see GUARD_INSTRUCTION_OPTIONS_1's module comment for why "hunter"
    isn't an option here).
    """
    feature_list = [
        [1.0 if i == j else 0.0 for i in range(len(GUARD_INSTRUCTION_OPTIONS_1))]
        for j in range(len(GUARD_INSTRUCTION_OPTIONS_1))
    ]
    active_policy = guard_instruction_split_1.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, GUARD_INSTRUCTION_OPTIONS_1)
    return chosen


GUARD_INSTRUCTION_OPTIONS = [
    f"{a}_{b}" for a in _TARGET_TYPES for b in _TARGET_TYPES
]

guard_instruction_policy = policy_utils.LinearPolicy(len(GUARD_INSTRUCTION_OPTIONS))

# Per-Day-1-board-config weight sets (2026-10 "split everything"
# extension -- see board_split.py's module docstring). Only asked once
# 2 Hunter claimants both need a watch slot assigned, which can only
# happen once CO claims exist -- well past Day 1's CO sequence -- so
# plain state.get("day1_config") is safe (no safe_day1_config() guard
# needed).
guard_instruction_split = board_split.BoardSplitManager(guard_instruction_policy)


def choose_guard_instruction_strategy(state):
    """
    Decide how to assign the two watch-target slots when 2 Hunter
    claimants both need one. Each slot independently gets one of
    "gray" (a random gray player), "medium" (the revealed Medium),
    "hunter" (the OTHER claimant), or "seer" (the revealed Seer).

    While EITHER of these two watch slots is active, the real
    Hunter's own guard choice is completely overridden (main.py's
    hunter_guard forces guard_instructions[true_hunter_idx] rather
    than calling hunter_ai.choose_guard_target) -- so for as long as
    a Hunter contest is unresolved, the Seer can ONLY ever be
    protected if one of these two slots is explicitly pointed at
    them. Before "seer" existed as its own category here, a claimed,
    living Seer -- ordinarily the single highest-value guard target
    in human play -- fell into the generic "gray" bucket along with
    every ordinary villager, so the village had no way to LEARN to
    prioritize protecting them during exactly the period (a fresh
    Hunter contest) they're often most exposed.

    Returns
    -------
    str
        One of the 16 "<slot0>_<slot1>" combinations in
        GUARD_INSTRUCTION_OPTIONS, e.g. "medium_seer".
    """
    feature_list = [
        [1.0 if i == j else 0.0 for i in range(len(GUARD_INSTRUCTION_OPTIONS))]
        for j in range(len(GUARD_INSTRUCTION_OPTIONS))
    ]
    active_policy = guard_instruction_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, GUARD_INSTRUCTION_OPTIONS)
    return chosen


# ============================================================
# 3-claimant version (2026-10-03 user request). Up to 2 wolves + 1
# Madman can all independently fake-claim "Hn" on top of the one real
# Hunter, so 3 simultaneous unresolved claimants is a real, reachable
# case -- the old 2-slot-only version above could only ever watch 2
# of them (main.py picked the 2 lowest player numbers, a seat-index
# bias with no strategic meaning), leaving a 3rd claimant completely
# untested until an earlier one resolved.
#
# Same flat, unconditional design as the 2-slot version, just with a
# 3rd independent slot: 4 target categories ^ 3 slots = 64 combinations.
# Kept as a SEPARATE policy (not a generalization of the 2-slot one)
# so the already-trained 2-slot weights for the common 2-claimant case
# are left completely untouched; this new 64-option policy starts
# fresh and only ever gets exercised (and trained) on the rarer
# 3-claimant boards.
# ============================================================

GUARD_INSTRUCTION_OPTIONS_3 = [
    "_".join(combo) for combo in itertools.product(_TARGET_TYPES, repeat=3)
]

guard_instruction_policy_3 = policy_utils.LinearPolicy(len(GUARD_INSTRUCTION_OPTIONS_3))
guard_instruction_split_3 = board_split.BoardSplitManager(guard_instruction_policy_3)


def choose_guard_instruction_strategy_3(state):
    """
    Same idea as choose_guard_instruction_strategy, but for 3
    simultaneous unresolved Hunter claimants: returns one of the 64
    "<slot0>_<slot1>_<slot2>" combinations in GUARD_INSTRUCTION_OPTIONS_3.
    """
    feature_list = [
        [1.0 if i == j else 0.0 for i in range(len(GUARD_INSTRUCTION_OPTIONS_3))]
        for j in range(len(GUARD_INSTRUCTION_OPTIONS_3))
    ]
    active_policy = guard_instruction_split_3.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, GUARD_INSTRUCTION_OPTIONS_3)
    return chosen


def _is_structurally_protected(state, candidate):
    """
    True if `candidate` is currently PROTECTED by a strong (but not
    absolutely certain) structural signal: a sole/uncontested claim of
    a claimable role, an independently confirmed villager (e.g. a
    night-attack victim or guard-verified survivor), or the confirmed
    real Hunter. Not a hard proof the way _is_hard_proven_fake_claim
    is -- an uncontested sole claimant COULD still be a wolf who
    simply never got contested -- but strong enough that treating it
    as a genuinely free choice for the learned policy is the wrong
    call. See apply_protection_floor for how this gets used.

    BUG FIX: a candidate whose OWN claim is itself hard-proven fake
    can never be "protected", full stop -- checked FIRST, before
    anything else. The "count" computation just below excludes hard-
    proven-fake claimants from the denominator so that the genuine
    sole survivor (once fakes are subtracted out) gets recognized and
    protected -- e.g. a late, structurally-impossible "Sy" claim
    shouldn't stop the REAL Day-1 Seer from being treated as sole.
    But without this early check, that same exclusion backfired on
    the FAKE claimant itself: computing "how many Sy claimants are
    left once fakes like me are removed" and getting 1 (the real
    Seer) was being read as "so count==1, I must be protected" --
    completely inverted, since the "1" it found was someone ELSE, not
    this candidate. That silently removed the most obviously-guilty
    candidates (a late Ps/Sy claim, score dozens of points above
    everyone else) from apply_protection_floor's candidate pool
    entirely, leaving the vote to fall on much weaker, ordinary
    suspects instead -- exactly the "why wasn't this obvious fake
    claimant ever executed" behavior reported from real play.
    """
    if _is_hard_proven_fake_claim(state, candidate):
        return False

    own_co = state["co_history"][candidate]
    if own_co is not None:
        # A claim already HARD-PROVEN fake (by an unconditional rule,
        # not a comparison to this candidate) doesn't get a vote in
        # whether this claim is "contested" -- see
        # _is_hard_proven_fake_claim. `candidate` itself is guaranteed
        # NOT hard-proven-fake at this point (see the early return
        # above), so it's never at risk of being the one excluded here.
        count = sum(
            1 for i, co in enumerate(state["co_history"])
            if co == own_co and not _is_hard_proven_fake_claim(state, i)
        )
        if count == 1:
            return True
    if candidate in state.get("confirmed_villagers", []):
        return True
    if candidate == state.get("confirmed_real_hunter"):
        return True
    return False


def apply_protection_floor(state, candidates):
    """
    Filter `candidates` down to the ones NOT currently structurally
    protected (see _is_structurally_protected), unless that would
    eliminate every candidate -- in which case the full, unfiltered
    list is returned instead (there's nobody left to point at
    otherwise). Shared by the village's own execution vote and the
    Medium's nomination-target choice: both are "who among these
    people looks worst" decisions, and neither should be able to land
    on someone the public record already all-but-clears, purely
    because a softmax over a still-small learned weight hasn't pushed
    their probability all the way down to zero yet. Comparing
    candidates against EACH OTHER (rather than against some externally
    computed "neutral" baseline) avoids the failure mode where every
    candidate looks "below neutral" for unrelated reasons and the
    floor silently disables itself.
    """
    # 2026-10-06 user request (information-asymmetry fix): this used to
    # HARD-EXCLUDE sole claimants / confirmed villagers / the confirmed
    # hunter from the village's vote pool. Wolves and the madman have
    # no such filter, so "voted for a protected player" became a
    # near-perfect non-villager oracle (92% of votes on sole claimants
    # came from wolf/madman in the logs) -- a rule, not something a
    # human villager could know. The pool is now the full candidate
    # list; avoiding such players is left to the learned weights
    # (is_sole, is_confirmed_villager, is_confirmed_real_hunter, ...),
    # exactly as it is for the wolf/madman side.
    return list(candidates)


def execution_distribution(state):
    """
    Same pool and scores as choose_execution_target, but returns the
    full {candidate: probability} distribution instead of sampling.
    Used by main.py to show a wolf where the village is likely to vote
    (wolf_ai's village_vote_prob vote feature). Pure: no sampling, no
    state changes.
    """
    candidates = state["candidates"]
    pool = apply_protection_floor(state, candidates)
    active_policy = execution_split.active(state.get("day1_config"))
    feature_list = [_extract_features(state, c) for c in pool]
    scores = [active_policy.score(f) for f in feature_list]
    probs = policy_utils.softmax(scores)
    dist = {c: 0.0 for c in candidates}
    for c, p in zip(pool, probs):
        dist[c] = p
    return dist


def choose_execution_target(state):
    """
    Decide which living player to execute during the day.

    Parameters
    ----------
    state : dict
        {
            "board": list[str],
            "co_history": list,
            "day": int,
            "candidates": list[int],   # living players (valid targets)
        }

    Returns
    -------
    int
        Index of the chosen execution target.
    """

    candidates = state["candidates"]
    feature_list = [_extract_features(state, c) for c in candidates]

    # NO HARD-EXECUTE RULE HERE ANYMORE. Before the Madman was added,
    # two situations were treated as unconditional, 100%-certain
    # "always execute this one" facts: a claim hard-proven fake by
    # game-rule logic (_is_hard_proven_fake_claim), and a confirmed-
    # genuine Seer's own "this player is a wolf" verdict (the "詰み"
    # checkmate, formerly state["seer_confirmed_wolf_accusations"]).
    # Both relied on the same underlying assumption: "only a wolf ever
    # fake-claims a role, or gets accused by a confirmed-non-wolf
    # Seer" -- so proving either one was equivalent to proving "this
    # is a wolf, remove it." A Madman breaks that equivalence: it can
    # fake-claim Ps/Hn/Sy exactly like a wolf, and can be confirmed
    # non-wolf (the confirmation never checks literal role) while
    # bluffing "Sy". So a proven-fake claim, or a confirmed-real-
    # Seer's accusation, now only proves "this claim/accusation is
    # fake" -- not "this specific person is a wolf". Worse, executing
    # a Madman isn't even unconditionally GOOD for the village the way
    # executing a wolf always is: config.py's comment on "Kg" explains
    # that removing a Madman shrinks `alive` without shrinking
    # `wolves`, which actually WORSENS the village's win-condition
    # margin (village = alive - wolves). Both signals are still
    # exactly as certain as ever at the "is this claim/accusation
    # fake" level, so they remain in the feature vector with boosted
    # LR/decay (is_fake, is_confirmed_contradicting_seer via
    # _CONFIRMED_TAIL_OFFSETS) -- what changed is that certainty no
    # longer bypasses the learned policy, since "therefore always
    # remove this player" is no longer automatically correct.
    active_policy = execution_split.active(state.get("day1_config"))
    scores = [active_policy.score(f) for f in feature_list]

    # SOFTER protection floor (not a hard rule -- an uncontested sole
    # claimant COULD still be a wolf who simply never got contested).
    # Relying purely on softmax sampling means even a strongly-
    # protective learned weight still gets picked sometimes by chance
    # -- e.g. a -1.46 gap against neutral candidates still means a
    # genuine ~5% chance of executing the sole claimant anyway,
    # observed directly in actual games. See apply_protection_floor's
    # docstring for why this compares candidates against each other
    # rather than an externally computed "neutral" baseline.
    pool = apply_protection_floor(state, candidates)
    feature_by_candidate = dict(zip(candidates, feature_list))
    pool_features = [feature_by_candidate[c] for c in pool]

    chosen, _probs = active_policy.choose(pool_features, pool)

    return chosen


# ============================================================
# Learning
# ============================================================

def update(kind, state, action, reward, learning_rate=0.01):
    """
    Apply one REINFORCE update for a single logged decision.

    `kind` / `state` / `action` come straight from an entry in
    main.py's `history` (the same shapes choose_execution_target
    was given / returned).
    """

    if kind == "execution":

        # BUG FIX: choose_execution_target never actually samples from
        # state["candidates"] (every living player) -- it narrows that
        # down to apply_protection_floor(state, candidates) FIRST, and
        # only THEN calls execution_policy.choose on the narrowed pool
        # (see choose_execution_target above). policy_utils.LinearPolicy
        # .update's own docstring is explicit that its feature_list/
        # candidates "must be the SAME candidate set the original
        # decision was made over ... so the gradient's baseline term is
        # correct" -- training on the full, unfiltered candidate list
        # here instead silently violated that requirement: the REINFORCE
        # baseline (expected_features, a probability-weighted average
        # over whatever candidate set is passed in) was being computed
        # over a DIFFERENT, larger distribution than the one execution_
        # policy.choose actually sampled `action` from, corrupting the
        # gradient for every feature in the vector, not just the ones
        # tied to protection.
        #
        # This stayed mostly harmless for a long time because apply_
        # protection_floor rarely excluded more than a candidate or two
        # -- the "full" and "narrowed" candidate sets were nearly
        # identical, so the wrong baseline was close enough to the
        # right one not to matter. It stopped being harmless the moment
        # _wolf_counting_deduction's Madman-counting fix (see main.py)
        # made confirmed_villagers trigger far more often and often
        # exclude MOST of the candidate set at once: a mismatch between
        # e.g. "sampled from 2 candidates" and "trained as if sampled
        # from 6" is a completely different animal from a mismatch of
        # 5 vs 6. Empirically, this is exactly what was driving the
        # post-fix training instability -- is_confirmed_villager's
        # weight (correctly, deliberately boosted-LR/decay -- see
        # _CONFIRMED_TAIL_OFFSETS) raced up to +49 in one especially
        # lopsided batch, and Ps_contest_wolf_prob / Sy_contest_wolf_
        # prob (otherwise well-behaved, previously-converged features
        # living in that SAME feature vector) collapsed to roughly -3.6
        # in that same batch -- collateral damage from a shared-vector
        # gradient computed against the wrong baseline, not a sign that
        # either the new deduction or contest_wolf_probability itself
        # is unsound.
        candidates = state["candidates"]
        pool = apply_protection_floor(state, candidates)
        feature_list = [_extract_features(state, c) for c in pool]

        active_policy = execution_split.active(state.get("day1_config"))
        active_policy.update(feature_list, pool, action, reward, learning_rate)
        execution_split.record_sample(state.get("day1_config"))

    elif kind == "guard_instruction_strategy_1":

        feature_list = [
            [1.0 if i == j else 0.0 for i in range(len(GUARD_INSTRUCTION_OPTIONS_1))]
            for j in range(len(GUARD_INSTRUCTION_OPTIONS_1))
        ]
        active_policy = guard_instruction_split_1.active(state.get("day1_config"))
        active_policy.update(
            feature_list, GUARD_INSTRUCTION_OPTIONS_1, action, reward, learning_rate
        )
        guard_instruction_split_1.record_sample(state.get("day1_config"))

    elif kind == "guard_instruction_strategy":

        feature_list = [
            [1.0 if i == j else 0.0 for i in range(len(GUARD_INSTRUCTION_OPTIONS))]
            for j in range(len(GUARD_INSTRUCTION_OPTIONS))
        ]
        active_policy = guard_instruction_split.active(state.get("day1_config"))
        active_policy.update(
            feature_list, GUARD_INSTRUCTION_OPTIONS, action, reward, learning_rate
        )
        guard_instruction_split.record_sample(state.get("day1_config"))

    elif kind == "guard_instruction_strategy_3":

        feature_list = [
            [1.0 if i == j else 0.0 for i in range(len(GUARD_INSTRUCTION_OPTIONS_3))]
            for j in range(len(GUARD_INSTRUCTION_OPTIONS_3))
        ]
        active_policy = guard_instruction_split_3.active(state.get("day1_config"))
        active_policy.update(
            feature_list, GUARD_INSTRUCTION_OPTIONS_3, action, reward, learning_rate
        )
        guard_instruction_split_3.record_sample(state.get("day1_config"))


# ============================================================
# Wolf-population estimation (public information only)
#
# STEP 4 of the Madman addition (see config.py's comment on "Kg"):
# the Madman must decide, in the endgame, whether to reveal itself to
# signal the wolves -- and that decision depends on how many wolves
# it believes are STILL ALIVE, even though the Madman has NO private
# knowledge of who the wolves actually are (confirmed by the user).
# A genuine Villager, asked to consider the symmetric "村騙り" bluff
# in the same endgame window, is in EXACTLY the same position -- so
# this lives here in village_ai.py (public-information-only
# territory, imported by both madman_ai.py and villager_ai.py) rather
# than in either side's own module.
# ============================================================

def estimate_wolf_alive_distribution(state):
    """
    Estimate P(exactly k wolves are currently ALIVE) for every
    k = 0..total_wolves, using only publicly available information.

    THE MODEL: every player who hasn't been independently confirmed
    non-wolf (state["confirmed_villagers"] / confirmed_real_hunter) or
    already confirmed a wolf (a Medium's "wolf" verdict, or a proven-
    fake Hunter claimant -- see the caveat below) is "unresolved" --
    could be a wolf, could not be. The game's true generative process
    (roles dealt uniformly at random, then narrowed by whoever
    happened to die or get investigated) is far too entangled with
    every side's actual PLAY to derive an exact posterior by hand. The
    same simplifying assumption already used for contest_wolf_
    probability is used again here: treat every unresolved player
    (dead or alive) as EQUALLY LIKELY to be hiding one of the
    remaining unaccounted-for wolves, ignoring any behavioral bias
    real play might introduce (e.g. wolves being somewhat more likely
    to survive to this point than a uniformly random villager would).
    That turns "how many of the unresolved wolves ended up among the
    still-alive unresolved players" into a textbook hypergeometric
    draw: W unresolved wolves and (U - W) unresolved non-wolves among
    U total unresolved players, of which A are currently alive --

        P(k unresolved wolves alive) =
            C(W, k) * C(U - W, A - k) / C(U, A)

    Any wolf ALREADY confirmed (by name) is added back in
    deterministically afterward if still alive -- their contribution
    isn't a probability, it's a known fact.

    CAVEAT this inherits from confirmed_dead_wolves elsewhere: a
    proven-fake Hunter claimant is treated as a confirmed wolf, which
    was airtight back when "only wolves ever fake-claim a role" but
    is now merely very likely (a Madman could be the fake claimant
    instead -- see village_ai.choose_execution_target's docstring).
    Rather than getting the wolf-population estimate itself tangled
    up in a second layer of Madman-vs-wolf uncertainty, this function
    keeps that existing approximation as-is; it's already the best
    signal main.py exposes without deeper rework, and this estimate
    is meant to be "good enough to inform a strategic gamble", not a
    certainty.

    Returns
    -------
    list[float]
        Length total_wolves + 1. Entry k is P(exactly k wolves alive).
    """
    total_wolves = state.get("total_wolves", 2)
    board = state.get("board", [])
    n_players = len(board)

    medium_reports = state.get("medium_reports", [])
    confirmed_fake_hunters = state.get("confirmed_fake_hunters", [])
    confirmed_wolves = set(r["player"] for r in medium_reports if r["is_wolf"])
    confirmed_wolves |= set(confirmed_fake_hunters)

    confirmed_non_wolf = set(state.get("confirmed_villagers", []))
    real_hunter = state.get("confirmed_real_hunter")
    if real_hunter is not None:
        confirmed_non_wolf.add(real_hunter)

    unresolved = [
        i for i in range(n_players)
        if i not in confirmed_wolves and i not in confirmed_non_wolf
    ]
    U = len(unresolved)
    W = max(0, total_wolves - len(confirmed_wolves))
    A = sum(1 for i in unresolved if board[i] != "--")

    confirmed_wolves_alive = sum(1 for i in confirmed_wolves if board[i] != "--")

    # Degenerate case: nothing left to be uncertain about (every wolf
    # already named, or no unresolved players remain at all) -- the
    # confirmed count alone is the whole distribution.
    dist = [0.0] * (total_wolves + 1)
    if U == 0 or W == 0:
        total_alive = min(confirmed_wolves_alive, total_wolves)
        dist[total_alive] = 1.0
        return dist

    W = min(W, U)
    lo = max(0, A - (U - W))
    hi = min(W, A)
    denom = math.comb(U, A)
    for k in range(lo, hi + 1):
        p = math.comb(W, k) * math.comb(U - W, A - k) / denom
        total_alive = min(confirmed_wolves_alive + k, total_wolves)
        dist[total_alive] += p

    return dist


TOTAL_WOLVES = config.ROLES.count("Wo")

NUM_ENDGAME_CONTEXT_FEATURES = (TOTAL_WOLVES + 1) + 2 + 1
# Shared context for the endgame "reveal / bluff" decision, used by
# BOTH sides of the same mechanic: the Madman's genuine reveal
# (madman_ai.choose_endgame_reveal_action) and a genuine Villager's
# "村騙り" decoy bluff (villager_ai.choose_endgame_bluff_action). Both
# are asked the exact same question ("should I add my name to
# self_declared_non_village right now?") from the exact same public
# vantage point -- neither one has any private knowledge the other
# lacks -- so this lives here once instead of being duplicated in
# both modules. See estimate_wolf_alive_distribution's docstring for
# [0..TOTAL_WOLVES], and each caller's own module for what the last 3
# entries mean for THEIR side specifically.


def endgame_context_features(state):
    """
    [0..TOTAL_WOLVES] : P(k wolves alive), k = 0..TOTAL_WOLVES (see
        estimate_wolf_alive_distribution).
    [-3] : is_endgame_3 -- exactly 3 players remain.
    [-2] : is_endgame_4 -- exactly 4 players remain.
    [-1] : already_has_role_claim -- state["self_index"]'s own
        co_history entry is one of CLAIMABLE_ROLES. For the Madman
        this means "I'm already bluffing Ps/Hn/Sy and would be giving
        that cover up entirely". For a genuine Villager this is
        almost always 0.0 (villager_day_co's general bluffing is
        disabled), but is computed the same way regardless so both
        callers share one implementation.
    """
    wolf_probs = estimate_wolf_alive_distribution(state)
    alive_count = state.get("alive_count", sum(1 for b in state["board"] if b != "--"))
    self_index = state.get("self_index")
    co_history = state["co_history"]
    already_has_role_claim = (
        1.0 if (self_index is not None and co_history[self_index] in CLAIMABLE_ROLES)
        else 0.0
    )
    return (
        list(wolf_probs)
        + [1.0 if alive_count == 3 else 0.0, 1.0 if alive_count == 4 else 0.0]
        + [already_has_role_claim]
    )
