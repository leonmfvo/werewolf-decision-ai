import itertools
import random

import policy_utils
import config
import village_ai
import board_split

CLAIMABLE_ROLES = config.CLAIMABLE_ROLES


# ============================================================
# Wolf AI
#
# All decision-making logic for the werewolf side. Each function
# receives only information the wolves are actually allowed to
# know (public board state + their own private wolf knowledge).
#
# Attack targeting and CO-strategy choice are linear-softmax
# policies (see policy_utils.LinearPolicy). Weights start at
# zero, so behaviour is currently identical to uniform random
# choice -- but is now backed by a real, learnable scorer.
# ============================================================


# --- Day-by-day CO decision: EACH living, unclaimed wolf ------
# independently chooses to claim one of the configured
# CLAIMABLE_ROLES, claim (plain) Villager, or stay silent. This
# is re-asked every day (see main.py's wolf_day_co), so wolves can
# REACT to what's already publicly known -- e.g. claim Hunter
# specifically because the real Hunter already revealed, to
# contest their claim and share in the village's "don't execute a
# Hunter claimant" leniency.
#
# GENERALIZATION: the option list is BUILT from config.CLAIMABLE_
# ROLES rather than hard-coded, so adding a new claimable role to
# config.py automatically gives wolves the option to fake-claim it
# too -- no changes needed here. Before ever fake-claiming a role,
# it's worth noting this only works because the option literally
# doesn't exist unless that role is actually part of the current
# composition (config.CLAIMABLE_ROLES) -- a wolf can't claim to be
# a role that was never dealt into this game.
#
# "claim_villager" WAS included as an experiment (to see if
# self-play would independently learn it's pointless, matching
# real human werewolf convention that a plain Villager has no
# ability worth "proving"). After training, it settled around
# 10-20% usage instead of fading toward zero -- not the clean
# self-correction hoped for -- so it's been removed rather than
# left in as dead weight every game.

CO_OPTIONS = [f"claim_{role}" for role in CLAIMABLE_ROLES] + ["stay_silent"]
NUM_OPTIONS = len(CO_OPTIONS)
# "has role X already been revealed?" for each claimable role,
# PLUS: "did an emergency (about-to-be-executed) reveal happen
# recently?" -- distinct from a routine reveal, since a claim made
# under that kind of pressure is a different signal (e.g. a wolf
# might specifically want to contest -- or avoid contesting -- a
# claim that came from someone visibly cornered) -- PLUS two more,
# mirroring hunter_ai's own under_threat/personally_nominated
# pattern: "am I MYSELF about to be executed right now" and "did
# today's nomination specifically name me". Without these two, a
# wolf choosing whether to claim here has no way to see that this
# particular decision is a last-second, nothing-to-lose gamble
# rather than a routine daily check -- it's asked from EXACTLY the
# same choose_co_action/CO_OPTIONS the routine check uses (see
# main.py's wolf_day_co vs. the emergency check in progression_ai),
# so these flags are what let the SAME policy tell the two moments
# apart and learn a different response to each -- PLUS one more set,
# "has one of MY OWN TEAMMATES (a private-knowledge fact, not
# something a villager could see) already claimed role X?", one flag
# per claimable role.
#
# This last set was added specifically because the plain "has role X
# been revealed" flags above are ANONYMOUS -- they fire identically
# whether role X was claimed by a teammate or by a real villager/
# Madman, so the CO decision itself had no way to tell "my partner
# already took Seer, I should back off / cover a different role"
# apart from "someone (possibly an enemy) already claimed Seer,
# maybe I shouldn't pile on". A real human werewolf team coordinates
# on WHO claims WHAT specifically to avoid two teammates colliding on
# the same role (which would force an unwinnable same-team contest)
# and to divide up which roles get covered -- that's a materially
# different signal from "avoid a role someone unrelated already
# claimed", and each (option, context) pair here gets its own learned
# weight (see the one-hot + interaction pattern below), so the two
# can now be told apart and weighted independently.
#
# UPDATE: Day 1 no longer uses this policy at all -- see
# DAY1_JOINT_CO_OPTIONS / choose_day1_joint_co_action below for why
# and what replaced it. In short: this policy's teammate_claimed_
# flags are always exactly 0 on Day 1 (build_public_state's `blind`
# flag zeroes out co_history entirely for a same-day glance, and
# REINFORCE's gradient is provably 0 whenever a feature is 0 for
# every candidate -- see policy_utils.LinearPolicy.update), so this
# policy could never have learned Day-1 teammate coordination in the
# first place no matter how it was trained. That doesn't mean real
# teammate coordination is impossible on Day 1 -- two actual wolves
# CAN agree on a joint plan overnight, they just can't react to each
# other DURING the simultaneous morning announcement itself, which is
# exactly what the new joint policy below models instead.
#
# From Day 2 onward, main.py still processes each side's CO decisions
# one player at a time and applies each claim to the shared board
# immediately, so a wolf asked LATER in that same day's pass
# genuinely can see (and react to) a teammate who already claimed
# earlier that same day -- exactly mirroring how the generic
# revealed_flags above already worked for same-day reactions, just
# now teammate-specific too. This policy remains exactly what
# main.py's wolf_day_co (Day 2+) and the separate emergency-evasion
# check use, both genuinely sequential/reactive moments where a
# teammate's already-public claim is real, checkable information.
#
# The Madman has no teammates in the current composition (config.py
# keeps exactly one "Kg"), so `state["teammate_indices"]` is simply
# empty whenever madman_ai reuses this same feature builder -- these
# flags then read as an unconditional all-zero for the Madman, which
# is the semantically correct answer ("I have nobody to coordinate
# with"), not a bug or an oversight.
NUM_CONTEXT_FEATURES = len(CLAIMABLE_ROLES) + 1 + 2 + len(CLAIMABLE_ROLES)

# One-hot option identity + context per option.
NUM_CO_FEATURES = NUM_OPTIONS + NUM_OPTIONS * NUM_CONTEXT_FEATURES

co_policy = policy_utils.LinearPolicy(NUM_CO_FEATURES)

# NOT split by board config (2026-10 user decision, chat discussion):
# unlike every other policy covered by the "split everything"
# extension, THIS decision (reveal a role / stay silent) is itself
# part of what PRODUCES the Day-1 board configuration in the first
# place -- splitting it BY that same configuration would be reasoning
# backwards from effect to cause, the same objection that already
# ruled out splitting wolf_ai.day1_joint_co_policy. Stays a single
# shared policy, exactly like before this extension (and exactly like
# hunter_ai.co_policy / madman_ai.co_policy, same reasoning).


def _co_feature_list(state):

    co_history = state["co_history"]
    revealed_flags = [1.0 if role in co_history else 0.0 for role in CLAIMABLE_ROLES]
    emergency_flag = 1.0 if state.get("emergency_reveals") else 0.0
    under_threat = 1.0 if state.get("under_threat", False) else 0.0
    personally_nominated = 1.0 if state.get("personally_nominated", False) else 0.0
    # WOLF/MADMAN-PRIVATE: has one of MY OWN teammates (not just
    # "someone", see the NUM_CONTEXT_FEATURES comment above) already
    # claimed role X? `teammate_indices` is supplied by main.py
    # (build_wolf_state-style private knowledge) and never includes
    # `self` -- empty for the Madman, who has no teammates to check.
    teammate_indices = state.get("teammate_indices", [])
    teammate_claimed_flags = [
        1.0 if any(co_history[t] == role for t in teammate_indices) else 0.0
        for role in CLAIMABLE_ROLES
    ]
    context_flags = (
        revealed_flags
        + [emergency_flag, under_threat, personally_nominated]
        + teammate_claimed_flags
    )

    features = []
    for j in range(NUM_OPTIONS):
        f = [0.0] * NUM_CO_FEATURES
        f[j] = 1.0
        for k, flag in enumerate(context_flags):
            f[NUM_OPTIONS + j * NUM_CONTEXT_FEATURES + k] = flag
        features.append(f)

    return features


def choose_co_action(state):
    """
    Decide whether to claim Medium, claim Hunter, or stay silent
    TODAY (only asked while this wolf is alive and hasn't claimed
    anything yet). Asked from two different moments that share this
    same policy, NEITHER of which is the routine Day-1 check anymore
    (see choose_day1_joint_co_action below for that): the routine
    daily check from Day 2 onward (main.py's wolf_day_co) and a
    last-second emergency evasion check (main.py's progression_ai,
    right before the vote, which can fire on ANY day including Day 1)
    for a wolf who has just become the leading execution target or
    been personally nominated -- a free, nothing-to-lose gamble a
    real cornered player would take.

    Parameters
    ----------
    state : dict
        Public state (board, co_history, day), optionally plus:
        "under_threat" : bool -- is THIS wolf about to be executed
            right now, before having claimed anything?
        "personally_nominated" : bool -- was THIS wolf specifically
            named in today's nomination?

    Returns
    -------
    str
        One of "claim_medium", "claim_hunter", "stay_silent"
    """

    feature_list = _co_feature_list(state)
    chosen, _probs = co_policy.choose(feature_list, CO_OPTIONS)

    return chosen


# --- Day 1 ONLY: joint two-wolf CO decision ---------------------
#
# Day 1 is a special case unlike any other CO decision in the game:
# both wolves are still alive (nobody can die before Day 1 in this
# game's flow), NOTHING has been publicly claimed by anyone yet, and
# -- per the user's explicit real-world framing -- a real wolf pair
# doesn't decide their morning claims independently and blind to each
# other; they can agree on a joint PLAN beforehand (an overnight wolf-
# only conversation), they just can't react moment-to-moment to each
# other DURING the simultaneous announcement itself. "Both stay
# silent", "one claims Seer, one claims Medium", "both claim Seer"
# (a mistake a real team would actively avoid) are all facets of ONE
# joint decision, not two independent ones -- so this is modeled as
# exactly that: a single choice over every UNORDERED pair of the 4
# per-wolf options (stay_silent + one claim per CLAIMABLE_ROLE),
# instead of two separate draws from co_policy above.
#
# UNORDERED, not all 16 ordered (option_A, option_B) combinations:
# the two wolves are interchangeable here (nothing in this game
# distinguishes "the physically-first-indexed wolf" from the other
# one -- no seat order, no speech-order effect), so which SPECIFIC
# wolf ends up claiming which role is irrelevant to the outcome, only
# the resulting COMBINATION is. Modeling it as 16 ordered options
# would need twice as many games to learn the same information (two
# ordered options mapping to the same unordered outcome), so
# itertools.combinations_with_replacement(CO_OPTIONS, 2) is used
# instead, giving exactly 10 options: 4 "both pick the same option"
# (including "both silent") + 6 "different pairs".
#
# Once a combination is chosen, main.py randomly decides which of the
# two actual wolves gets which half of the pair (when they differ) --
# see main.py's wolf_day1_joint_co -- so the model never has to learn
# an arbitrary, meaningless preference for "player-index order" that
# has nothing to do with strategy.
#
# NO CONTEXT FEATURES: unlike every other CO-style decision in this
# file, this one takes no state at all. Day 1, before either wolf has
# acted, has no varying public information to condition on -- there
# is nothing for a context flag to describe yet (see this section's
# own reasoning above: revealed_*/emergency_* would all be 0 anyway,
# exactly like they always were for Day-1 draws from co_policy). This
# policy is free to be exactly what it looks like: a single learned
# PRIOR distribution over the 10 combinations, reflecting nothing
# more than which joint plan tends to win, on average, across all of
# self-play -- which is really all Day 1 ever had to go on anyway.
DAY1_JOINT_CO_OPTIONS = list(itertools.combinations_with_replacement(CO_OPTIONS, 2))
NUM_DAY1_JOINT_CO_OPTIONS = len(DAY1_JOINT_CO_OPTIONS)

day1_joint_co_policy = policy_utils.LinearPolicy(NUM_DAY1_JOINT_CO_OPTIONS)


def _day1_joint_co_feature_list():
    """Plain one-hot identity per combination -- see this section's own comment for why no context is needed."""
    return [
        [1.0 if i == j else 0.0 for i in range(NUM_DAY1_JOINT_CO_OPTIONS)]
        for j in range(NUM_DAY1_JOINT_CO_OPTIONS)
    ]


def choose_day1_joint_co_action(state=None):
    """
    Decide, ONCE per game, what the two wolves will jointly claim on
    Day 1. `state` is accepted (and ignored) only so this fits the
    same choose_*(state) calling convention every other decision in
    this file uses -- see this section's module comment for why no
    context actually applies here.

    Returns
    -------
    tuple[str, str]
        One of DAY1_JOINT_CO_OPTIONS, e.g. ("claim_Sy", "stay_silent")
        -- unordered: main.py decides which physical wolf gets which
        half.
    """
    feature_list = _day1_joint_co_feature_list()
    chosen, _probs = day1_joint_co_policy.choose(feature_list, DAY1_JOINT_CO_OPTIONS)
    return chosen


# --- Attack target ---------------------------------------------

NUM_ATTACK_FEATURES = 21  # 2026-10-06: +[14] is_confirmed_villager, [15] late*is_gray
# [0] bias
# [1] is_psychic_claimant        -- this target has claimed Psychic
# [2] is_known_real_psychic      -- WOLF-PRIVATE: by elimination, any
#                                    Psychic claimant who isn't one of
#                                    our own wolves must be the real one
# [3] is_known_hunter            -- this target has claimed Hunter.
#                                    NOTE: unlike is_known_real_psychic,
#                                    this does NOT exclude our own
#                                    teammates -- but that's harmless
#                                    here specifically because
#                                    werewolf_attack's own candidate
#                                    list already excludes every living
#                                    wolf, so a teammate's fake Hunter
#                                    claim can never itself be a valid
#                                    attack target in the first place.
# [4] day (normalized)
# [5] exposes_teammate_rival_claim -- WOLF-PRIVATE: attacking this
#     target counts as a night-attack death, which unconditionally
#     confirms them non-wolf the moment it happens (see main.py's
#     attack_victims). If a wolf TEAMMATE is currently claiming the
#     SAME role as this target, that exact confirmation immediately
#     proves the teammate's own claim fake too (see village_ai's
#     _is_confirmed_fake_by_cleared_rival -- only a wolf ever
#     fake-claims a role here, so clearing one contested claimant
#     convicts the other). Killing a genuine threat (a real Hunter,
#     say) is valuable, but doing it by attacking the specific
#     claimant a teammate is actively contesting can cost that
#     teammate's cover in the very same stroke.
# [6] is_known_real_seer         -- WOLF-PRIVATE: by the exact same
#     elimination logic as is_known_real_psychic, any Seer claimant
#     who isn't one of our own wolves must be the real one. The real
#     Seer is arguably the single most dangerous ongoing threat once
#     revealed (a live, proactive source of new information every
#     single night, unlike the Medium's one-shot-per-execution
#     report), so this deserves its own explicit signal rather than
#     being lumped in with is_psychic_claimant/is_known_hunter.
# [7] exposes_teammate_rival_claim * is_known_real_psychic
# [8] exposes_teammate_rival_claim * is_known_hunter
# [9] exposes_teammate_rival_claim * is_known_real_seer
#     WOLF-PRIVATE interaction terms -- [5] alone forces a SINGLE,
#     role-blind trade-off between "kill this known threat" and
#     "don't burn a teammate's contested claim of the same role",
#     because a plain linear model has no other way to let that
#     trade-off differ by WHICH role is being contested: the combined
#     effect of [5] and (say) [6] together is always just their sum,
#     whatever the two individual weights happen to be. In practice
#     the right trade-off genuinely differs by role -- killing a
#     newly-revealed real Seer is worth far more than killing a real
#     Hunter or Medium (see is_known_real_seer's weight, which trains
#     to roughly 3-5x the others), so the same flat exposure penalty
#     that appropriately deters attacking a contested Medium/Hunter
#     claimant is nowhere near enough to deter attacking a contested
#     SEER claimant -- the seer-kill priority simply swamps it. These
#     three explicit products let training learn a SEPARATE penalty
#     for "this exposure cost, specifically when the target being
#     killed is the confirmed-real Psychic / Hunter / Seer", instead
#     of being stuck with whatever the sum of two independently-tuned
#     weights happens to produce.
# [10] is_cleared_white_by_known_real_seer -- WOLF-PRIVATE: some Sy
#     claimant who (by the same elimination logic as
#     is_known_real_seer -- not one of our own wolves, so must be
#     genuine) has PUBLICLY reported this exact target as NOT a wolf
#     (main.py's seer_reports). Attacking this target doesn't cost a
#     teammate's cover the way [5]/[7-9] do, but it's still a
#     meaningfully DIFFERENT kind of target than an un-reported one:
#     killing someone the real Seer already vouched for burns a night
#     on a player the village already trusts (no suspicion to gain by
#     removing them), whereas an un-vouched-for target might still be
#     hiding real suspicion the wolves would rather leave alone to
#     fester. Kept separate from is_known_real_seer (which is about
#     attacking the SEER themself) since these are two entirely
#     different targets with different strategic value.
# [11] guard_risk_active * is_known_hunter -- a living Hn claimant
#     (able to guard) AND a confirmed-real Sy claimant (worth
#     protecting) both exist right now. This is the standard human
#     Werewolf "kill the Hunter first" logic: the Hunter cannot guard
#     THEMSELF (hunter_ai.choose_guard_target's candidates already
#     exclude self), so attacking a Hunter claimant is NEVER at risk
#     of being blocked by a guard, and doing so permanently removes
#     the one thing that could otherwise save the real Seer on some
#     future night. Without this feature, [3] (is_known_hunter) and
#     [6] (is_known_real_seer) are scored independently, and since a
#     newly-revealed real Seer is normally the single most valuable
#     kill (see [6]'s comment -- it settles roughly 1.5x higher than
#     [3] on its own), the policy has no way to learn "go for the
#     SAFER guaranteed kill first, specifically because a guardable
#     Seer exists" -- it just always prefers whichever target scores
#     higher in isolation, seer included, guard risk and all.
# [12] guard_risk_active * is_known_real_seer -- same guard_risk_active
#     condition, but attached to the SEER target instead: attacking a
#     confirmed-real Seer while a Hunter claimant is still alive to
#     possibly guard them is a wasted-night risk that [6] alone
#     doesn't capture (a plain linear model can only give the Seer one
#     flat priority value, not a LOWER one specifically while the
#     guard threat is still live). Expected to learn negative,
#     working opposite [11] to shift the preferred order toward
#     "Hunter now, Seer once the guard is gone" instead of "whichever
#     of the two scores higher this instant."
#
# guard_risk_active itself (not a standalone feature, only ever used
# multiplied into [11]/[12]) = 1.0 iff some living player has claimed
# Hn (co_history == "Hn", still alive -- a genuine guard threat as far
# as the wolves can tell, whether or not that claimant turns out to be
# real) AND some known-real Sy claimant exists (candidate in
# known_real_seers, from [6]'s own computation) worth protecting.
# [13] is_announced_guard_target (2026-10-04 user request): this exact
#     candidate is one of TONIGHT's publicly announced guard-
#     instruction targets (state["guard_instructions"].values() -- the
#     same "護衛指示" line main.py prints every day once a Hunter claim
#     is unresolved). This is PUBLIC information (unlike guard_risk_
#     active's coarser "a guard threat structurally exists somewhere"
#     flag, which never looks at WHICH specific player got watched) --
#     an attack on this specific target has some chance of being
#     blocked outright if the claimant who named them happens to be
#     the real Hunter, something guard_risk_active (which only ever
#     reasons about is_known_hunter/is_known_real_seer) cannot express
#     at all when the watched target is neither of those two (e.g. a
#     "gray" or "medium" watch slot). Deliberately a SOFT, learned
#     signal, not a hard exclusion: with 2-3 claimants, UP TO 3 watch
#     slots exist and only ONE of them (if any) belongs to the real
#     Hunter, so most announced targets aren't actually protected --
#     avoiding ALL of them whenever one is a high-value kill (a real
#     Seer, say) would often be overcautious.

# The attack decision gets its OWN, more aggressive learning dynamics
# than the rest of the wolf side's policies. Empirically, the shared
# defaults (learning_rate=0.03, decay_rate=0.03) were too timid here:
# is_known_hunter's weight plateaued around +1.5 to +1.9 (only ~60%
# attack probability with a handful of targets, ~37% actual kill
# rate on a revealed Hunter in a 500-game trace) instead of the near-
# certain "always finish the Hunter once they're known" behavior that
# should be strictly optimal (a live, revealed Hunter is the single
# biggest threat to the wolves; there's no real downside feature
# modeled for killing them). The cause: this signal only shows up a
# couple of times per game, so its REINFORCE gradient is small and
# gets matched almost exactly by the constant per-epoch decay once
# the weight is "confident enough" -- an equilibrium, not a bug in
# the individual feature.
#
# Simply lowering decay_rate globally does fix this (weight climbs
# to ~2.9, win rate ~34%) but reopens the EXACT instability the
# value_function switch was meant to close: vote_policy's is_teammate
# weight (equally rare, but noisier) flipped positive again --
# wolves voting FOR their own teammate -- within a few hundred
# epochs at a lower global decay. So the two rare wolf signals need
# opposite treatment: attack's is_known_hunter needs LESS regularization
# to fully commit, vote's is_teammate needs the standard (or more)
# regularization to stay stable. Hence a per-policy multiplier
# instead of touching the shared constants in train_and_report.py.
#
# Verified side-by-side (900 epochs / 45,000 games each): with these
# multipliers, is_known_hunter settles around +3.0 to +3.6 (a revealed
# Hunter becomes the overwhelming preferred target) while vote_policy's
# is_teammate weight stays consistently negative (-0.7 to -1.3, same
# as the unmodified default), and the trustworthy 1000-game village
# win rate came down to ~35% (vs. ~43-46% with the shared defaults).
ATTACK_LR_MULTIPLIER = 1.5
ATTACK_DECAY_MULTIPLIER = 0.35

# Per-feature boost on TOP of the whole-policy ATTACK_LR/DECAY_MULTIPLIER
# above, for indices [7],[8],[9] only (the three interaction terms).
#
# Measured after adding the interaction terms (22,805 epochs / 1.14M
# games): index [9] (exposes_teammate_rival_claim * is_known_real_seer)
# settled at -0.001 -- essentially untouched -- while the two features
# it's built from settled at is_known_real_seer=+4.576 and
# exposes_teammate_rival_claim=-0.338. Behaviorally this showed up as
# no real change either: in contested-Sy games, the real Seer was still
# getting confirmed (attacked) by game's end at ~58-62% both before and
# after adding the raw interaction features.
#
# The cause is the same "rare signal vs. constant decay" equilibrium
# already documented above for is_known_hunter, but worse: that feature
# needs only "this target claimed Hunter", while an interaction term
# needs its narrower CONJUNCTION -- "target is a confirmed-real
# claimant" AND "a teammate is independently contesting that exact
# role" -- to co-occur in the same game. It fires far less often than
# is_known_hunter did, so even the whole-policy 1.5x/0.35x multipliers
# that were enough to unstick is_known_hunter aren't enough here; the
# per-epoch decay keeps erasing the tiny gradient before it can
# accumulate into a meaningful value.
_ATTACK_LR_MULTIPLIERS = [1.0] * NUM_ATTACK_FEATURES
_ATTACK_DECAY_MULTIPLIERS = [1.0] * NUM_ATTACK_FEATURES
_INTERACTION_LR_BOOST = 6.0
_INTERACTION_DECAY_CUT = 0.15
# [11]/[12] (guard_risk_active * is_known_hunter / is_known_real_seer)
# are the SAME "narrow conjunction, rare co-occurrence" situation
# documented above for [7]/[8]/[9] -- guard_risk_active itself already
# needs a living Hn claimant AND a confirmed-real Sy claimant to
# co-occur, so they get the identical boost rather than being left to
# the whole-policy default and risking the same decay-outpaces-
# gradient equilibrium.
#
for _idx in (7, 8, 9, 11, 12):
    _ATTACK_LR_MULTIPLIERS[_idx] = _INTERACTION_LR_BOOST
    _ATTACK_DECAY_MULTIPLIERS[_idx] = _INTERACTION_DECAY_CUT

# [2]/[3]/[6] (is_known_real_psychic / is_known_hunter / is_known_real
# _seer THEMSELVES, not just their interaction terms above) get a
# DECAY-ONLY relief, added after a real game log showed the wolves
# attacking a plain, uninvolved villager over a sole, UNGUARDED
# (guard_risk_active == 0 -- the real Hunter was already dead)
# confirmed-real Seer who went on to find a wolf teammate black the
# very next investigation. is_known_real_seer's own weight, measured
# at +4.576 back when [7]/[8]/[9] were first added (see that comment),
# had drifted all the way to -1.422 by the time this was caught, and
# is_known_hunter (documented above as needing the whole-policy
# 1.5x/0.35x boost just to reach +3.0-3.6) had similarly decayed back
# down to +0.390 -- both consistent with the interaction terms above,
# once THEY were boosted, soaking up the gradient credit for "the
# known-real claim was worth attacking" whenever an interaction also
# applied, leaving the base feature to keep losing ground to ordinary
# decay on every game where it didn't (guard_risk_active == 0 is
# common -- the real Hunter dies or never claims plenty of games).
#
# The first attempt at fixing this reused the SAME 6x LR / 0.15x decay
# tier as the interaction terms above -- but unlike those NARROW
# conjunctions, [2]/[3]/[6] are ALREADY common (some claimant of each
# role exists most games), so a 6x learning-rate multiplier on top of
# already-frequent updates didn't fix the equilibrium, it blew past
# it: a 600-epoch trial run swung is_known_real_psychic between -2.2
# and +9.1 and pushed is_known_real_seer to -5 to -6, WORSE than the
# -1.422 this was meant to fix (verified by re-running the same trial
# and inspecting the weight trajectory every 100 epochs -- it never
# settled). The actual problem here was never "too little gradient
# when the feature fires" (unlike the genuinely rare interaction
# terms) -- it was "decay erodes the weight between the frequent times
# it fires just enough to matter" -- so only the DECAY side needs
# correcting; stacking the interaction terms' aggressive LR boost on
# top of an already-frequent feature just amplifies ordinary batch-to-
# batch noise into destructive oscillation instead of convergence.
for _idx in (2, 3, 6):
    _ATTACK_DECAY_MULTIPLIERS[_idx] = _INTERACTION_DECAY_CUT

attack_policy = policy_utils.LinearPolicy(
    NUM_ATTACK_FEATURES,
    lr_multipliers=_ATTACK_LR_MULTIPLIERS,
    decay_multipliers=_ATTACK_DECAY_MULTIPLIERS,
)

# Per-Day-1-board-config weight sets for the night attack target (see
# board_split.py). attack_policy above stays the shared/common policy
# exactly as before -- attack_split.active(day1_config) is what
# actually decides/learns from this point on; it's attack_policy
# itself until train_and_report.enable_board_split() is called.
attack_split = board_split.BoardSplitManager(attack_policy)


def _extract_attack_features(state, candidate):

    co_history = state["co_history"]
    day = state["day"]
    wolf_indices = state["wolf_indices"]

    is_ps_claimant = 1.0 if co_history[candidate] == "Ps" else 0.0

    ps_claimants = [i for i, co in enumerate(co_history) if co == "Ps"]
    known_real_psychics = [i for i in ps_claimants if i not in wolf_indices]
    is_known_real_psychic = 1.0 if candidate in known_real_psychics else 0.0

    is_known_hunter = 1.0 if co_history[candidate] == "Hn" else 0.0

    sy_claimants = [i for i, co in enumerate(co_history) if co == "Sy"]
    known_real_seers = [i for i in sy_claimants if i not in wolf_indices]
    is_known_real_seer = 1.0 if candidate in known_real_seers else 0.0

    seer_reports = state.get("seer_reports", [])
    is_cleared_white_by_known_real_seer = 1.0 if any(
        r["player"] == candidate and not r["is_wolf"] and r["claimant"] in known_real_seers
        for r in seer_reports
    ) else 0.0

    candidate_co = co_history[candidate]
    exposes_teammate_rival_claim = 0.0
    if candidate_co in CLAIMABLE_ROLES:
        for teammate in wolf_indices:
            if teammate != candidate and co_history[teammate] == candidate_co:
                exposes_teammate_rival_claim = 1.0
                break

    board = state["board"]
    hunter_claimant_alive = any(
        co_history[i] == "Hn" and board[i] != "--" for i in range(len(co_history))
    )
    guard_risk_active = 1.0 if hunter_claimant_alive and known_real_seers else 0.0

    guard_instructions = state.get("guard_instructions", {})
    is_announced_guard_target = 1.0 if candidate in guard_instructions.values() else 0.0

    return [
        1.0,
        is_ps_claimant,
        is_known_real_psychic,
        is_known_hunter,
        day / 7.0,
        exposes_teammate_rival_claim,
        is_known_real_seer,
        exposes_teammate_rival_claim * is_known_real_psychic,
        exposes_teammate_rival_claim * is_known_hunter,
        exposes_teammate_rival_claim * is_known_real_seer,
        is_cleared_white_by_known_real_seer,
        guard_risk_active * is_known_hunter,
        guard_risk_active * is_known_real_seer,
        is_announced_guard_target,
        # 2026-10-06 additions
        1.0 if candidate in state.get("confirmed_villagers", []) else 0.0,
        (1.0 if sum(1 for b in board if b != "--") <= 5 else 0.0)
        * (1.0 if candidate_co is None else 0.0),
        *_h5_attack_features(state, candidate, wolf_indices, known_real_seers),
    ]


def _h5_attack_features(state, candidate, wolf_indices, sy_nonwolf):
    """2026-10-06 (H5): [16] a wolf teammate fake-blacked this candidate
    (killing them proves the teammate's black false), [17] a teammate
    fake-whited them, [18] they were saved by a GJ before, [19] the
    candidate is a Sy claimant who whited one of our wolves (almost
    surely the Madman), [20] the candidate is a Sy claimant who blacked
    one of our wolves (almost surely the real Seer)."""
    reports = state.get("seer_reports", [])
    f16 = 1.0 if any(r["player"] == candidate and r["is_wolf"]
                     and r["claimant"] in wolf_indices for r in reports) else 0.0
    f17 = 1.0 if any(r["player"] == candidate and not r["is_wolf"]
                     and r["claimant"] in wolf_indices for r in reports) else 0.0
    f18 = 1.0 if candidate in state.get("guard_success_targets", []) else 0.0
    f19 = 1.0 if candidate in sy_nonwolf and any(
        r["claimant"] == candidate and r["player"] in wolf_indices and not r["is_wolf"]
        for r in reports) else 0.0
    f20 = 1.0 if candidate in sy_nonwolf and any(
        r["claimant"] == candidate and r["player"] in wolf_indices and r["is_wolf"]
        for r in reports) else 0.0
    return [f16, f17, f18, f19, f20]


def choose_attack_target(state):
    """
    Decide which living non-wolf player to attack at night.

    Parameters
    ----------
    state : dict
        {
            "board": list[str],
            "co_history": list,
            "day": int,
            "wolf_indices": list[int],
            "targets": list[int],
        }

    Returns
    -------
    int
        Index of the chosen attack target.
    """

    targets = state["targets"]
    feature_list = [_extract_attack_features(state, t) for t in targets]

    active_policy = attack_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, targets)

    return chosen


# --- Day vote target ---------------------------------------------
#
# Execution is no longer decided by a single "village consensus"
# pick -- EVERY living player, wolves included, independently casts
# one vote, and the plurality is executed (see main.py's run_vote).
# A wolf's vote has a completely different motive from a genuine
# villager's: it isn't "who do I suspect", it's "who can I get away
# with voting for" -- above all, never give away a wolf teammate.
# That's why this needs its OWN policy (village_ai's execution_policy
# is shared by every non-wolf voter, since they're all reasoning
# from the same honest, public-information viewpoint -- but a wolf's
# viewpoint is fundamentally different and includes PRIVATE
# knowledge, is_wolf_teammate below, that no villager could ever
# have).

NUM_VOTE_FEATURES = 15  # 2026-10-06: +[10..14], see _extract_vote_features
# [0] bias
# [1] is_wolf_teammate     -- WOLF-PRIVATE: this candidate is one of
#                              our own -- should learn a strongly
#                              negative weight (never expose a
#                              teammate) once training has anything
#                              to go on.
# [2] is_psychic_claimant  -- claimed Ps (voting for a claimed role
#                              looks more "justified" to onlookers
#                              than picking on a silent gray).
# [3] is_hunter_claimant   -- claimed Hn.
# [4] is_confirmed_villager -- publicly, unambiguously safe already;
#                              voting here just wastes the vote (and
#                              looks odd) with nothing to gain.
# [5] day (normalized)
# [6] is_sole_trustworthy_nomination_target -- this candidate was
#     named today by an UNCONTESTED claimant of a claimable role
#     (e.g. the sole Medium claimant's "指定"). The village side
#     (village_ai's defied_sole_nomination_count feature) treats
#     NOT voting for this player as a suspicious "defiance" signal
#     -- so a wolf who wants to blend in has a real incentive to
#     comply and vote here too, exactly like a real player would.
#     Without this feature the wolf side had no way to even SEE
#     which candidate was the trustworthy nomination target, so it
#     could never learn to react to that pressure at all.
# [7] is_seer_claimant     -- claimed Sy. Kept separate from
#     is_psychic_claimant/is_hunter_claimant (rather than folded into
#     a generic "claimed something" flag) so training can learn
#     whatever caution a live Seer accusation specifically deserves
#     independently of the other two roles' own dynamics.
# [8] is_self_declared_non_village -- STEP 4/5 of the Madman addition
#     (config.py's "Kg" comment): this candidate has publicly declared
#     themselves hostile to the village (main.py's
#     self_declared_non_village), via either the Madman's genuine
#     endgame reveal OR a genuine Villager's "村騙り" decoy bluff (see
#     villager_ai.choose_endgame_bluff_action -- that's exactly what
#     keeps this feature from being a hard certainty even for a wolf).
#     A wolf doesn't know who the Madman is any more than anyone else
#     does, so this is an ordinary LEARNED suspicion feature here, not
#     a hard "never vote for a known ally" exclusion -- but a wolf
#     that wants the Madman's coordination signal to actually pay off
#     has every reason to let this feature learn a strongly negative
#     weight (steer away from voting for a fellow self-declared
#     claimant) once training gives it something to go on.
# [9] village_vote_prob -- 2026-10-05 user request: how likely a
#     villager, reasoning from the SAME public information, is to vote
#     for this candidate (village_ai.execution_distribution, scaled
#     relative to a uniform vote and capped; state["village_vote_probs"]
#     is filled by main.py right before this decision). Lets a wolf see
#     where the village is heading, so it can learn to blend in with
#     the likely vote, or to pull a vote away from a teammate the
#     village is about to pick.

vote_policy = policy_utils.LinearPolicy(NUM_VOTE_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 extension). This
# decision only ever happens during the day vote, strictly after Day
# 1's CO sequence has finished, so plain state.get("day1_config") is
# safe here (no safe_day1_config() guard needed -- see that function's
# docstring for which decisions DO need it).
vote_split = board_split.BoardSplitManager(vote_policy)


def _is_sole_trustworthy_nomination_target(state, candidate):
    co_history = state["co_history"]
    nominations = state.get("nominations", {})

    claim_counts = {}
    for co in co_history:
        if co in CLAIMABLE_ROLES:
            claim_counts[co] = claim_counts.get(co, 0) + 1

    for nominator, target in nominations.items():
        if target != candidate:
            continue
        nominator_co = co_history[nominator]
        if nominator_co not in CLAIMABLE_ROLES:
            continue
        if claim_counts.get(nominator_co, 0) > 1:
            continue  # nominator's claim is contested -- not trustworthy
        return True
    return False


def _village_vote_prob_feature(state, candidate):
    probs = state.get("village_vote_probs") or {}
    n = len(probs)
    if n == 0:
        return 0.0
    return min(probs.get(candidate, 0.0) * n, 4.0) / 2.0


def _extract_vote_features(state, candidate):

    co_history = state["co_history"]
    day = state["day"]
    wolf_indices = state.get("wolf_indices", [])

    is_wolf_teammate = 1.0 if candidate in wolf_indices else 0.0
    is_ps_claimant = 1.0 if co_history[candidate] == "Ps" else 0.0
    is_hn_claimant = 1.0 if co_history[candidate] == "Hn" else 0.0
    is_sy_claimant = 1.0 if co_history[candidate] == "Sy" else 0.0
    is_confirmed_villager = 1.0 if candidate in state.get("confirmed_villagers", []) else 0.0
    is_sole_trustworthy_nomination_target = (
        1.0 if _is_sole_trustworthy_nomination_target(state, candidate) else 0.0
    )
    is_self_declared_non_village = (
        1.0 if candidate in state.get("self_declared_non_village", []) else 0.0
    )

    return [
        1.0,
        is_wolf_teammate,
        is_ps_claimant,
        is_hn_claimant,
        is_confirmed_villager,
        day / 7.0,
        is_sole_trustworthy_nomination_target,
        is_sy_claimant,
        is_self_declared_non_village,
        _village_vote_prob_feature(state, candidate),
        *_extra_vote_features(state, candidate, wolf_indices),
    ]


def _extra_vote_features(state, candidate, wolf_indices):
    """2026-10-06: [10] this wolf's OWN fake black on the candidate
    (stay consistent with its claim), [11] a teammate's fake black on
    the candidate, [12] late(alive<=5)*is_claimant, [13] final(alive<=3)
    * village_vote_prob (blend in / pull votes in the last days), [14]
    is_sole_claimant (the village no longer filters these out, so a
    wolf has to be able to see them)."""
    co_history = state["co_history"]
    voter = state.get("voter")
    reports = state.get("seer_reports", [])
    own_black = 1.0 if any(
        r["claimant"] == voter and r["player"] == candidate and r["is_wolf"]
        for r in reports) else 0.0
    mate_black = 1.0 if any(
        r["claimant"] in wolf_indices and r["claimant"] != voter
        and r["player"] == candidate and r["is_wolf"]
        for r in reports) else 0.0
    alive = sum(1 for b in state["board"] if b != "--")
    co = co_history[candidate]
    is_claimant = 1.0 if co in CLAIMABLE_ROLES else 0.0
    is_sole = 1.0 if (co in CLAIMABLE_ROLES and
                      sum(1 for c in co_history if c == co) == 1) else 0.0
    return [
        own_black,
        mate_black,
        (1.0 if alive <= 5 else 0.0) * is_claimant,
        (1.0 if alive <= 3 else 0.0) * _village_vote_prob_feature(state, candidate),
        is_sole,
    ]


def choose_vote_target(state):
    """
    Decide who this (one specific, living) wolf votes to execute
    today.

    Parameters
    ----------
    state : dict
        Same shape as build_wolf_state's output, PLUS
        "vote_candidates": list[int] -- every OTHER living player
        (nobody votes for themselves).

    Returns
    -------
    int
        Index of the player this wolf votes for.
    """

    candidates = state["vote_candidates"]
    feature_list = [_extract_vote_features(state, c) for c in candidates]

    active_policy = vote_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, candidates)

    return chosen


# ============================================================
# Endgame decoy declaration ("捨て身の偽装CO"): once only 3 or 4
# players remain, a living wolf can OPENLY declare itself hostile to
# the village, exactly like madman_ai.choose_endgame_reveal_action and
# villager_ai.choose_endgame_bluff_action already can -- writing into
# the exact same main.py self_declared_non_village list they do (see
# main.py's wolf_endgame_declare).
#
# WHY A WOLF WOULD EVER WANT THIS: unlike the Madman's reveal (whose
# point is to help the remaining WOLVES coordinate their own vote) or
# the Villager's bluff (whose point is to make the Madman's genuine
# reveal ambiguous), a wolf declaring itself hostile isn't signaling
# to a teammate or protecting someone else's cover -- it's PURE SELF-
# DEFENSE against a structural hole the user identified from a real
# played-out game: before this option existed, self_declared_non_
# village could ONLY ever be populated by a genuine Madman or a
# genuine Villager, never by an actual wolf. So the moment every OTHER
# living player chose to declare (which training was already pushing
# genuine Villagers to do quite often -- see villager_ai's endgame
# bluff), the one player who did NOT declare was trivially, perfectly
# identifiable as the wolf by simple elimination -- no suspicion
# feature or learned weight even needed, just counting. Giving the
# wolf the same action closes that hole: once a wolf can ALSO show up
# in self_declared_non_village, "did this player declare?" stops being
# a wolf-identifying signal on its own, for anyone, restoring the
# ambiguity the whole self_declared_non_village mechanic exists for.
#
# This is a genuinely defensive, cover-blending move ONLY in the
# specific sense that it removes the "silent lone holdout" tell -- the
# wolf is not lying about anything (unlike the Villager's bluff, this
# declaration is TRUE), so it carries the same real risk the Madman's
# own reveal does (converging suspicion once alive_count is this low),
# which is exactly why this is a REAL LEARNED decision (mirroring
# madman_ai's approach) rather than a hard "always declare" rule --
# training has to discover whether blending in this way is actually
# worth it, and under what board conditions.
# Four options, not two: as well as staying silent, a wolf can openly
# CLAIM any one of the three team identities -- "村人CO" (claim
# village, the subtler play: reassure everyone else without admitting
# hostility), "狼CO" (claim wolf -- the ORIGINAL "declare_hostile"
# option this replaced, true self-outing), or "狂人CO" (claim madman,
# poaching the Madman's own cover). See main.py's endgame_claims for
# how these get tracked publicly and village_ai's is_sole_kg_claimant/
# is_contested_kg_claimant/is_confirmed_contradicting_kg_claim for how
# a CONTESTED "狂人CO" claim (at most one real Madman exists, so 2+
# simultaneous claimants means at least one is provably lying) turns
# into exactly the same kind of hard deduction already used for a
# contested Ps/Hn/Sy claim.
ENDGAME_DECLARE_OPTIONS = ["stay_silent", "claim_village", "claim_wolf", "claim_madman"]

# One-hot option identity + context*option interaction terms, exactly
# mirroring madman_ai._endgame_reveal_feature_list's shape (and, in
# turn, seer_ai._co_feature_list's) -- see village_ai.endgame_context_
# features for what each context value means. NOT a single shared
# feature vector scored identically for every option, which would
# leave nothing for training to ever learn between them.
NUM_ENDGAME_DECLARE_OPTIONS = len(ENDGAME_DECLARE_OPTIONS)
NUM_ENDGAME_DECLARE_FEATURES = (
    NUM_ENDGAME_DECLARE_OPTIONS
    + NUM_ENDGAME_DECLARE_OPTIONS * village_ai.NUM_ENDGAME_CONTEXT_FEATURES
)

endgame_declare_policy = policy_utils.LinearPolicy(NUM_ENDGAME_DECLARE_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 extension). Only ever
# asked once alive_count is 3-4, long after Day 1 -- plain
# state.get("day1_config") is safe.
endgame_declare_split = board_split.BoardSplitManager(endgame_declare_policy)


def _endgame_declare_feature_list(state):
    context = village_ai.endgame_context_features(state)
    features = []
    for j in range(NUM_ENDGAME_DECLARE_OPTIONS):
        f = [0.0] * NUM_ENDGAME_DECLARE_FEATURES
        f[j] = 1.0
        for k, value in enumerate(context):
            f[NUM_ENDGAME_DECLARE_OPTIONS + k * NUM_ENDGAME_DECLARE_OPTIONS + j] = value
        features.append(f)
    return features


def choose_endgame_declare_action(state):
    """
    Decide whether to openly claim an identity (village/wolf/madman)
    RIGHT NOW, or stay silent (only ever asked while alive_count is 3
    or 4 -- see main.py's wolf_endgame_declare).

    Parameters
    ----------
    state : dict
        Public state (main.py's build_public_state) plus
        "self_index" (this wolf's own board index, so
        already_has_role_claim can be computed) and "alive_count".

    Returns
    -------
    str
        One of ENDGAME_DECLARE_OPTIONS
    """
    feature_list = _endgame_declare_feature_list(state)
    active_policy = endgame_declare_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, ENDGAME_DECLARE_OPTIONS)
    return chosen


# ============================================================
# Learning
# ============================================================

def update(kind, state, action, reward, learning_rate=0.01):
    """
    Apply one REINFORCE update for a single logged decision.

    `kind` / `state` / `action` come straight from an entry in
    main.py's `history`.
    """

    if kind == "co_action":

        feature_list = _co_feature_list(state)
        co_policy.update(feature_list, CO_OPTIONS, action, reward, learning_rate)

    elif kind == "day1_joint_co":

        feature_list = _day1_joint_co_feature_list()
        day1_joint_co_policy.update(
            feature_list, DAY1_JOINT_CO_OPTIONS, action, reward, learning_rate
        )

    elif kind == "attack":

        targets = state["targets"]
        feature_list = [_extract_attack_features(state, t) for t in targets]

        active_policy = attack_split.active(state.get("day1_config"))
        active_policy.update(feature_list, targets, action, reward,
                              learning_rate * ATTACK_LR_MULTIPLIER)
        attack_split.record_sample(state.get("day1_config"))

    elif kind == "vote_target":

        candidates = state["vote_candidates"]
        feature_list = [_extract_vote_features(state, c) for c in candidates]

        active_policy = vote_split.active(state.get("day1_config"))
        active_policy.update(feature_list, candidates, action, reward, learning_rate)
        vote_split.record_sample(state.get("day1_config"))

    elif kind == "fake_declaration_strategy":

        feature_list = fake_declaration_feature_list(state)
        active_policy = fake_declaration_split.active(state.get("day1_config"))
        active_policy.update(feature_list, FAKE_DECLARATION_OPTIONS, action, reward, learning_rate)
        fake_declaration_split.record_sample(state.get("day1_config"))

    elif kind == "fake_report_strategy":

        feature_list = fake_report_feature_list(state)
        active_policy = fake_report_split.active(state.get("day1_config"))
        active_policy.update(feature_list, FAKE_REPORT_OPTIONS, action, reward, learning_rate)
        fake_report_split.record_sample(state.get("day1_config"))

    elif kind == "fake_seer_report_strategy":

        # Must reconstruct the EXACT SAME candidate set choose_fake_
        # seer_report_strategy sampled `action` from (see LinearPolicy.
        # update's docstring on why a mismatched candidate set gives a
        # wrong baseline) -- which is the narrower Day-1-only list on
        # Day 1, and the full list on every other day.
        options = (
            FAKE_SEER_REPORT_DAY1_OPTIONS if state.get("day") == 1
            else FAKE_SEER_REPORT_OPTIONS
        )
        option_indices = [FAKE_SEER_REPORT_OPTIONS.index(o) for o in options]
        feature_list = [
            [1.0 if i == idx else 0.0 for i in range(len(FAKE_SEER_REPORT_OPTIONS))]
            for idx in option_indices
        ]
        active_policy = fake_seer_report_split.active(state.get("day1_config"))
        active_policy.update(feature_list, options, action, reward, learning_rate)
        fake_seer_report_split.record_sample(state.get("day1_config"))

    elif kind == "fake_seer_accuse_target":

        candidates = state["candidates"]
        feature_list = _accuse_target_feature_list(state)
        active_policy = accuse_target_split.active(state.get("day1_config"))
        active_policy.update(feature_list, candidates, action, reward, learning_rate)
        accuse_target_split.record_sample(state.get("day1_config"))

    elif kind == "endgame_declare":

        feature_list = _endgame_declare_feature_list(state)
        active_policy = endgame_declare_split.active(state.get("day1_config"))
        active_policy.update(feature_list, ENDGAME_DECLARE_OPTIONS, action, reward, learning_rate)
        endgame_declare_split.record_sample(state.get("day1_config"))


# ============================================================
# Fake white-declaration strategy (when a wolf fake-claims a
# role that would let the real holder vouch for someone as
# "confirmed white" -- e.g. claiming Hunter and claiming to have
# guarded someone). Since a fake claimant has no REAL guard data,
# whatever they "declare" is a lie -- but the village can't tell
# that yet, only after the guard-instruction verification resolves
# who's genuine. This decides WHO the lie targets.
# ============================================================

FAKE_DECLARATION_OPTIONS = ["declare_teammate", "declare_random"]

# 2026-10-06 (M3): context features, attached to the "declare_teammate"
# option only (the other option is the zero baseline). state["fake_decl_ctx"]
# = [other_hn_alive, guard_success_seen, teammate_alive, day/7, late,
#    own_prev_teammate_declares].
FAKE_DECL_CTX_N = 6
NUM_FAKE_DECL_FEATURES = len(FAKE_DECLARATION_OPTIONS) + FAKE_DECL_CTX_N


def fake_declaration_feature_list(state):
    ctx = list(state.get("fake_decl_ctx") or [0.0] * FAKE_DECL_CTX_N)
    rows = []
    for j in range(len(FAKE_DECLARATION_OPTIONS)):
        row = [1.0 if i == j else 0.0 for i in range(len(FAKE_DECLARATION_OPTIONS))]
        row += ctx if FAKE_DECLARATION_OPTIONS[j] == "declare_teammate" else [0.0] * FAKE_DECL_CTX_N
        rows.append(row)
    return rows


fake_declaration_policy = policy_utils.LinearPolicy(NUM_FAKE_DECL_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 extension). Only ever
# asked once a Hunter contest + verification is underway (well past
# Day 1's CO sequence) -- plain state.get("day1_config") is safe.
fake_declaration_split = board_split.BoardSplitManager(fake_declaration_policy)


def choose_fake_declaration_strategy(state):
    """
    Decide who a FAKE Hunter claim should falsely vouch for as
    "confirmed white": the wolf's own teammate (protecting them,
    if believed) or a random other living player.

    Returns
    -------
    str
        One of "declare_teammate", "declare_random"
    """
    feature_list = fake_declaration_feature_list(state)
    active_policy = fake_declaration_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, FAKE_DECLARATION_OPTIONS)
    return chosen


# ============================================================
# Fake Medium report strategy: a wolf claiming "Ps" has no real
# ability, but each day the REAL Medium reports on yesterday's
# execution, the fake one can EITHER back up that verdict (looks
# consistent, but gives the village nothing to work with) or
# CONTRADICT it (creates a "split" -- valuable-looking information
# that risks getting the liar caught once the truth becomes clear
# some other way, e.g. next execution's outcome).
# ============================================================

FAKE_REPORT_OPTIONS = ["agree", "contradict"]

# 2026-10-06 (M3): context features, attached to "contradict" only
# (agree = zero baseline). state["fake_ctx"] =
# [truth_is_wolf (wolf only), other_ps_alive, other_ps_prev_black,
#  own_prev_black, executed_was_claimant, teammate_ps (wolf only),
#  late, day/7]. NO same-day verdict of another claimant is ever
# included: all morning reports are simultaneous.
FAKE_REPORT_CTX_N = 8
NUM_FAKE_REPORT_FEATURES = len(FAKE_REPORT_OPTIONS) + FAKE_REPORT_CTX_N


def fake_report_feature_list(state):
    ctx = list(state.get("fake_ctx") or [0.0] * FAKE_REPORT_CTX_N)
    rows = []
    for j in range(len(FAKE_REPORT_OPTIONS)):
        row = [1.0 if i == j else 0.0 for i in range(len(FAKE_REPORT_OPTIONS))]
        row += ctx if FAKE_REPORT_OPTIONS[j] == "contradict" else [0.0] * FAKE_REPORT_CTX_N
        rows.append(row)
    return rows


fake_report_policy = policy_utils.LinearPolicy(NUM_FAKE_REPORT_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 extension). Only ever
# asked the day AFTER an execution has happened -- well past Day 1's
# CO sequence -- so plain state.get("day1_config") is safe.
fake_report_split = board_split.BoardSplitManager(fake_report_policy)


def choose_fake_report_strategy(state):
    """
    Decide whether a fake Medium claim's daily report should agree
    with or contradict the real Medium's verdict on yesterday's
    execution.

    Returns
    -------
    str
        One of "agree", "contradict"
    """
    feature_list = fake_report_feature_list(state)
    active_policy = fake_report_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, FAKE_REPORT_OPTIONS)
    return chosen


# ============================================================
# Fake Seer report strategy: a wolf claiming "Sy" has no real
# investigative ability, but has to produce a daily report just like
# the genuine Seer would (main.py's fake_seer_report). Unlike a fake
# Medium's report (which only ever needs to agree/contradict a
# single, already-known verdict about yesterday's execution), a fake
# Seer has to invent BOTH a target and a verdict for a fabricated
# "investigation" every day -- a much richer lie with real strategic
# options:
#   - accuse a RIVAL Seer claimant of being a wolf: risky (openly
#     hostile), but potentially very valuable -- if this liar is ever
#     mistaken for the confirmed-real Seer (village_ai's checkmate
#     rule), it can get a genuine threat auto-executed.
#   - vouch for a TEAMMATE as clean: protects that teammate's cover,
#     the same motive as the existing fake-Hunter declaration.
#   - accuse or clear a random uninvolved player: the safe, low-
#     information default.
# ============================================================

FAKE_SEER_REPORT_OPTIONS = [
    "accuse_rival_black", "vouch_teammate_white", "random_black", "random_white",
]

# DAY 1 ONLY: a claimant's very first public report is always standing
# in for the mandatory pre-game "random white" investigation (main.py's
# _assign_seer_random_white) -- the real Seer is handed a guaranteed
# NON-wolf target before Day 1 even starts, with no night action
# involved, so a genuine Seer's Day-1 report can NEVER come back
# black. A fake claimant reporting black on Day 1 would therefore be
# just as instantly, structurally self-incriminating as an impossible
# late claim (village_ai's _is_impossible_late_forced_claim) -- so the
# two black-verdict options are simply not offered as choices that day,
# rather than trusting the (day-blind -- see below) learned policy to
# discover on its own that it should never pick them. From Day 2
# onward a fabricated report is about a genuine NIGHT investigation
# slot, which the real Seer's own free choice could equally have come
# back black on, so the full option list applies again from there.
FAKE_SEER_REPORT_DAY1_OPTIONS = ["vouch_teammate_white", "random_white"]

fake_seer_report_policy = policy_utils.LinearPolicy(len(FAKE_SEER_REPORT_OPTIONS))

# Per-Day-1-board-config weight sets (2026-10 extension). Even the
# Day-1 instance of this report is produced AFTER Day 1's whole CO
# sequence has settled (it stands in for the mandatory pre-game
# investigation, reported alongside the claim itself, not mid-
# sequence) -- plain state.get("day1_config") is safe.
fake_seer_report_split = board_split.BoardSplitManager(fake_seer_report_policy)


def choose_fake_seer_report_strategy(state):
    """
    Decide what today's fabricated Seer "investigation result" should
    be: who it names, and whether it calls them a wolf or clears them
    -- see main.py's fake_seer_report for how each option resolves to
    an actual (target, is_wolf) pair.

    On Day 1 the candidate list is narrowed to FAKE_SEER_REPORT_DAY1_
    OPTIONS (see its comment for why) -- note this is done by
    restricting the CANDIDATE SET passed to the shared policy's
    choose(), not by adding a "day" feature to it: fake_seer_report_
    policy's feature vectors are plain one-hot option identities with
    no day information at all, so its learned preference between (say)
    "vouch_teammate_white" and "random_white" is a single global
    number that applies on every day equally -- only WHICH options are
    even offered varies by day, decided here structurally rather than
    learned.

    Returns
    -------
    str
        One of FAKE_SEER_REPORT_OPTIONS (or, on Day 1, the narrower
        FAKE_SEER_REPORT_DAY1_OPTIONS).
    """
    options = (
        FAKE_SEER_REPORT_DAY1_OPTIONS if state.get("day") == 1
        else FAKE_SEER_REPORT_OPTIONS
    )
    option_indices = [FAKE_SEER_REPORT_OPTIONS.index(o) for o in options]
    feature_list = [
        [1.0 if i == idx else 0.0 for i in range(len(FAKE_SEER_REPORT_OPTIONS))]
        for idx in option_indices
    ]
    active_policy = fake_seer_report_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, options)
    return chosen


# ============================================================
# "accuse_rival_black" TARGET selection -- WHICH rival Sy claimant to
# name, once the strategy above has already decided to accuse someone.
#
# Previously this was hard-coded as `random.choice(pool)` in main.py's
# fake_seer_report: no feature, no learning, no way to tell "this
# rival is probably the real Seer" apart from "this rival is anyone
# else". A wolf's own teammate who ALSO claimed Sy was even included
# in that same random pool, so a real, no-benefit-at-all mistake
# (slandering your own side) happened by pure chance as often as
# slandering the genuine threat. main.py now excludes wolf teammates
# from this pool structurally before it ever reaches here (there is no
# scenario where accusing your own teammate helps -- that's a strict
# dominance, not something worth spending learning signal on), so
# every candidate `choose_accuse_target` sees is a genuine
# non-teammate rival: either the real Seer, or (if one exists and also
# claimed Sy) the Madman. Telling those two apart isn't always
# possible with certainty, so which one is the better target is
# exactly what training should discover.
# ============================================================

NUM_ACCUSE_TARGET_FEATURES = 12  # 2026-10-06: +[5..11], used for EVERY fake-seer target choice


def _accuse_target_feature_list(state):
    """
    Shared by wolf_ai and madman_ai (madman_ai imports this function
    directly -- it has no teammates to exclude, but the target-side
    reasoning is otherwise identical: "of the remaining non-teammate
    Sy claimants, which one is worth accusing?").

    state["candidates"] : the pool to choose among (already had
        teammates removed by main.py's fake_seer_report, for a wolf).
    """
    candidates = state["candidates"]
    claim_day = state.get("claim_day") or []
    confirmed_villagers = state.get("confirmed_villagers", [])
    alive_count = state.get("alive_count", 0)
    is_sole_candidate = 1.0 if len(candidates) == 1 else 0.0

    co_history = state.get("co_history") or []
    wolf_indices = state.get("wolf_indices", [])
    seer_reports = state.get("seer_reports", [])
    probs = state.get("village_vote_probs", {})
    n = max(1, len(co_history))
    black = 1.0 if state.get("fake_polarity") else 0.0
    white = 1.0 - black if "fake_polarity" in state else 0.0
    features = []
    for c in candidates:
        cd = claim_day[c] if c < len(claim_day) else None
        is_gray = 1.0 if (co_history and co_history[c] is None) else 0.0
        is_claimant = 1.0 if (co_history and co_history[c] in CLAIMABLE_ROLES) else 0.0
        reported_by_other = 1.0 if any(
            r["player"] == c and r["claimant"] != state.get("self_index")
            for r in seer_reports) else 0.0
        exec_prob = min(probs.get(c, 0.0) * n, 4.0) / 2.0
        features.append([
            1.0,
            1.0 if c in confirmed_villagers else 0.0,
            (cd / 7.0) if cd else 0.0,
            is_sole_candidate,
            alive_count / config.TOTAL_PLAYERS,
            # 2026-10-06: the target of a fake report is now a learned
            # choice for every strategy (it used to be random.choice):
            is_gray,
            1.0 if c in wolf_indices else 0.0,
            reported_by_other,
            exec_prob,
            black * is_gray,
            black * exec_prob,
            white * is_claimant,
        ])
    return features


accuse_target_policy = policy_utils.LinearPolicy(NUM_ACCUSE_TARGET_FEATURES)

# Per-Day-1-board-config weight sets (2026-10 extension). Only ever
# reached from within fake_seer_report_strategy's "accuse_rival_black"
# branch, which is itself safe to key on day1_config directly (see
# fake_seer_report_split's comment) -- same reasoning applies here.
accuse_target_split = board_split.BoardSplitManager(accuse_target_policy)


def choose_accuse_target(state):
    """
    Pick WHICH candidate in state["candidates"] to name in an
    "accuse_rival_black" fake Seer report. Only ever called when
    len(candidates) > 1 (main.py skips straight to random.choice --
    behaviourally identical to this policy's own zero-init softmax --
    when there's nothing to actually decide).
    """
    feature_list = _accuse_target_feature_list(state)
    active_policy = accuse_target_split.active(state.get("day1_config"))
    chosen, _probs = active_policy.choose(feature_list, state["candidates"])
    return chosen
