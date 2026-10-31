"""
Shared infrastructure for "graduating" a policy from ONE set of
weights shared across every game, to separate weight sets keyed by
the game's Day-1 CO configuration -- how many players claimed each of
Sy(占い師)/Ps(霊能者)/Hn(狩人) on Day 1, e.g. (2,2,1).

Why this exists: a single shared linear policy has exactly ONE weight
per feature, for the whole game -- it can't also make that weight
depend on WHICH board configuration produced the feature. But some
decisions genuinely call for different strategies depending on the
board's shape (e.g. whether to bite a Hunter claimant differs between
a 1-1-1 and a 1-1-3 board; which contested claim to roll first can
matter differently at 2-3 than at 2-2). That's a multiplicative
(config x feature) interaction a linear model can't represent no
matter how much data it sees -- this is a genuine capacity limit, not
a "hasn't converged yet" problem. Splitting the weights themselves,
per config, is the most direct fix (see the chat discussion this
shipped with, 2026-09, for the alternatives considered and rejected).

As of the 2026-10 "split everything" extension, this wraps every
eligible decision-making policy across every AI module (see each
module's own `*_split = board_split.BoardSplitManager(...)` lines).
Two kinds of decision are deliberately left OUT, both for the same
reason -- they are part of what PRODUCES day1_config in the first
place, so splitting them BY day1_config would be reasoning backwards
from effect to cause:
  - wolf_ai.day1_joint_co_policy (the two wolves' joint Day-1 claim)
  - the three "reveal a role or stay silent" timing policies --
    wolf_ai.co_policy, hunter_ai.co_policy, madman_ai.co_policy
    (2026-10 user decision, chat discussion) -- even their Day-2+/
    emergency-evasion calls, after day1_config has already settled,
    are still part of the SAME kind of decision that shapes the board,
    so these stay fully shared across the board, not just on Day 1.
The shared value_function.py critic is also left unsplit, on separate
grounds: it's meant to estimate expected outcome uniformly across
every side/decision-type, not per board config.
"""

import policy_utils


# (Sy, Ps, Hn) Day-1 claim counts. These 8 cover ~92% of games
# (measured over a 20,000-game sample against the live trained
# policy, 2026-09) -- PLUS (3,2,0) and (2,3,0), individually rarer
# (~0.3-0.4% each) but kept as their OWN buckets anyway, because they
# were the two low-village-win-rate configurations that motivated this
# whole feature in the first place; lumping them into OTHER_BUCKET
# would defeat the point of building this at all. Every other
# combination (~20 more, each individually rarer than either of
# those two) shares OTHER_BUCKET so they don't each get a
# near-permanently-starved weight set of their own.
COMMON_DAY1_CONFIGS = [
    (2, 2, 1), (2, 2, 0), (1, 2, 1), (1, 3, 1),
    (1, 2, 2), (1, 2, 0), (1, 3, 0), (2, 1, 1),
    (3, 2, 0), (2, 3, 0),
]
_COMMON_SET = set(COMMON_DAY1_CONFIGS)
OTHER_BUCKET = "other"

# One-way, EXPERIMENTAL widening of the split -- see enable_full_split()
# below. False (default): only COMMON_DAY1_CONFIGS get their own
# bucket, everything else shares OTHER_BUCKET. True: every distinct
# (Sy,Ps,Hn) combination seen gets its own bucket, no grouping at all.
FULL_SPLIT_MODE = False


def enable_full_split():
    """
    Widen the split from "the ~10 common configs plus one shared
    OTHER_BUCKET" (the default) to give EVERY distinct Day-1 board
    configuration its own bucket. This is a SEPARATE, ALSO one-way
    switch from BoardSplitManager.enable_split() -- calling
    enable_split() alone still groups rare configs into OTHER_BUCKET;
    this widens that further.

    Deliberately experimental (see the chat discussion this shipped
    with, 2026-09): the whole point of grouping rare configs into
    OTHER_BUCKET in the first place was that some of them are SO rare
    (the tail of day1_config_report's distribution has entries below
    0.05% of games) that a bucket of their own would take millions of
    additional games to reach even a few hundred on-policy samples --
    use this to measure that cost directly, and how much (if
    anything) fully separating them actually buys, rather than taking
    either number on faith.

    Safe to call at any point after BoardSplitManager.enable_split()
    has been called on a manager (nothing happens to a manager that's
    still in shared mode until enable_split() is ALSO called on it).
    A board config that already has its own bucket is unaffected; one
    that was sharing OTHER_BUCKET gets its own bucket the next time
    that config is played, warm-started from OTHER_BUCKET's CURRENT
    weights (not the older frozen shared base) -- see
    BoardSplitManager.active's own comment for why that's the better
    starting point at this stage.

    2026-10-02: this global flag is now only the DEFAULT for a
    manager whose own `force_full_split()` has never been called (see
    BoardSplitManager.force_full_split) -- a manager pinned to True or
    False via that method ignores this flag entirely. Flipping this
    flag therefore widens only the managers still following the
    global default, not necessarily "every split policy".
    """
    global FULL_SPLIT_MODE
    FULL_SPLIT_MODE = True


def bucket_key(day1_config, full=None):
    """
    day1_config: a (sy, ps, hn) tuple (main.py's _current_day1_config()),
    or None/missing -- treated as OTHER_BUCKET rather than raising, so
    a caller that hasn't wired a "day1_config" key into its state dict
    yet degrades to the shared "other" bucket instead of crashing.

    full: whether to use FULL_SPLIT_MODE-style "every distinct config
    gets its own bucket" behavior for THIS call. None (the default)
    falls back to the global FULL_SPLIT_MODE flag -- this is what every
    caller used before 2026-10's per-manager override existed, and
    what BoardSplitManager instances that don't set their own
    `full_split_override` still get. Pass True/False explicitly to
    override the global flag for one particular manager -- see
    BoardSplitManager.force_full_split.
    """
    if day1_config is None:
        return OTHER_BUCKET
    effective_full = FULL_SPLIT_MODE if full is None else full
    if effective_full:
        return day1_config
    if day1_config in _COMMON_SET:
        return day1_config
    return OTHER_BUCKET


def format_day1_config(day1_config):
    """
    Render a (sy, ps, hn) claim-count tuple as a plain claimant-count
    string, e.g. (1, 2, 1) -> "1-2-1", (2, 2, 2) -> "2-2-2",
    (3, 1, 0) -> "3-1-0".

    Each of the three占/霊/狩 slots is rendered independently from its
    own count, with NO role name attached (the slot ORDER -- 占-霊-狩,
    i.e. Sy-Ps-Hn -- is what carries which slot is which; the caller is
    expected to label that order once, e.g. in a column header):
      - n claimants (n >= 0) -> the digit n itself, e.g. "0", "1", "2"
        (0 included -- nobody claiming this role is just that role's
        count being zero, same as any other count)

    (2026-10-04 user request: this used to render each count as a
    role-claim-composition string instead, e.g. "真狼-真狂-真狼" --
    but that notation reads as near-identical to, and is shown right
    alongside, the DETAILED true-composition breakdown this bucket
    label is paired with elsewhere (train_and_report.
    _format_top_detailed_day1_configs), which already spells out who's
    genuine/Werewolf/Madman per slot -- e.g. "2-2-2 計482件:
    真狼-真狂-真狼". Having the bare COUNT label echo that same 真/狼
    vocabulary made the two look like duplicates of each other instead
    of a count paired with its breakdown, so this reverted to plain
    digits for the count-only label; the detailed string is still the
    one that uses 真/狼/狂. Later the same day, the 0-count case was
    switched from "欠" to plain "0" too, for the same reason --
    "3-1-欠" mixing a digit notation with one leftover kanji read as
    inconsistent; "3-1-0" is uniformly digits like every other count.)

    day1_config=None (e.g. no Day-1 config settled yet) renders as
    "?-?-?" rather than raising, matching bucket_key's own None
    handling above.
    """
    if day1_config is None:
        return "?-?-?"

    def _slot(n):
        return str(max(n, 0))

    sy, ps, hn = day1_config
    return f"{_slot(sy)}-{_slot(ps)}-{_slot(hn)}"


def format_day1_config_detailed(day1_claims, true_roles):
    """
    Like format_day1_config, but for displaying ONE ACTUAL GAME whose
    true roles are known (main.py's ROLES) -- 2026-10-02 follow-up to
    the user's original notation request, after they pointed out that
    a real game's "真狼-真狼-真狼" was wrong when one of those "狼"
    slots was actually the Madman claiming, not a Werewolf ("真狂" is
    the correct rendering for a genuine-plus-Madman-fake pair).

    format_day1_config (above) ONLY ever sees a bare (sy, ps, hn)
    COUNT -- used by board_split.py's bucket routing, and by every
    AGGREGATE report in train_and_report.py that summarizes thousands
    of games sharing the same (sy,ps,hn) count (which, across all
    those different games, can each have a totally different real/
    Werewolf/Madman makeup -- there's no single "true" breakdown to
    show for a whole bucket of games, so those call sites correctly
    keep using the generic, faction-blind "真"/"狼" count notation).
    THIS function is for the one place that's displaying a SINGLE,
    already-finished game (main.py's Day-1 CO announcement) and so
    genuinely has access to who each claimant actually is.

    Parameters
    ----------
    day1_claims : list of (player_idx, claimed_role_code)
        Every Day-1 CO claim made this game, in the order they
        happened (main.py's _day1_co_display_buffer) -- claimed_role_
        code is one of config.CLAIMABLE_ROLES ("Sy"/"Ps"/"Hn").
    true_roles : list indexed by player_idx
        The game's real role assignment (main.py's ROLES) -- used to
        classify each claimant as the genuine holder ("真"), a
        Werewolf faking it ("狼"), or the Madman faking it ("狂").

    For each of the three占/霊/狩 slots, independently:
      - no claimants -> "欠"
      - the genuine holder, if they claimed, is always rendered FIRST
        as "真", regardless of the actual claim order (matching
        format_day1_config's existing "真 always leads" convention) --
        then every OTHER claimant of that same role follows, each
        rendered as "狼" (a Werewolf) or "狂" (the Madman) in the order
        they actually claimed. A slot with no genuine claimant at all
        (the real holder died or stayed silent before any fake beat
        them to it) is rendered as just its faker(s), e.g. "狼" or
        "狼狂", with no "真" -- this can't happen with the Day-1-only
        claims this function is meant for (campaigning this early,
        every specialist is still alive and has had the chance to
        claim), but is handled rather than assumed away.
      - an unexpected true role for a claimant (should not happen in
        the current ruleset -- only the real holder, a Werewolf, or
        the Madman can ever claim a specialist role) renders as "?"
        rather than silently mislabeling or crashing.
    """
    order = ["Sy", "Ps", "Hn"]
    by_role = {role: [] for role in order}
    for player_idx, claimed_role in day1_claims:
        if claimed_role not in by_role:
            continue
        true_role = true_roles[player_idx]
        if true_role == claimed_role:
            symbol = "真"
        elif true_role == "Wo":
            symbol = "狼"
        elif true_role == "Kg":
            symbol = "狂"
        else:
            symbol = "?"
        by_role[claimed_role].append(symbol)

    def _slot(symbols):
        if not symbols:
            return "欠"
        # genuine claimant (if any) always leads, matching
        # format_day1_config's "真 first" convention; everyone else
        # keeps their actual claim order.
        real = [s for s in symbols if s == "真"]
        others = [s for s in symbols if s != "真"]
        return "".join(real + others)

    return "-".join(_slot(by_role[role]) for role in order)


class BoardSplitManager:
    """
    Wraps an EXISTING policy_utils.LinearPolicy (`base_policy`) so
    every call site that already reads/writes that object directly
    (train_and_report.py's weight snapshots, persistence.py's save/
    load, self_play.py's decay) keeps working completely unchanged --
    `base_policy` stays the shared/common policy, forever.

    What THIS adds is `.active(day1_config)`: the policy that should
    actually be used for the CURRENT decision. Before `.enable_split()`
    is ever called, that's always just `base_policy` -- zero behavior
    change from before this existed. After `.enable_split()`, it
    becomes a per-bucket LinearPolicy, WARM-STARTED (on first use) as
    a copy of `base_policy`'s weights at the moment of the switch --
    so a bucket only has to learn how THIS board should differ from
    the already-converged shared baseline, not relearn everything
    from zero. That's what makes even OTHER_BUCKET-level rare
    configurations viable at all: without warm-starting, a full
    from-scratch split would need roughly (number of buckets) times
    the training data to reach the same per-bucket confidence the
    shared policy already has, and the very buckets that most need
    help (the rare, low-win-rate ones) would be starved hardest.

    `base_policy` is frozen (no more gradient updates, no more decay)
    from the moment `.enable_split()` is called, so it stays one
    stable, common ancestor that every bucket -- including ones not
    touched until much later -- warm-starts from identically.
    """

    def __init__(self, base_policy):
        self.base = base_policy
        self.split_enabled = False
        self.buckets = {}  # bucket_key -> LinearPolicy
        # bucket_key -> cumulative count of TRAINING samples (REINFORCE
        # updates) that bucket has ever received, across every
        # train()/save_weights() run since the split was enabled --
        # NOT reset by a single evaluation batch or report run, unlike
        # day1_config_report's per-call counts. This is what answers
        # "how much has this bucket actually learned from so far" --
        # see record_sample().
        self.sample_counts = {}
        # bucket_key -> set, accumulating every key touched by
        # record_sample() since the last decay_all() call. This is
        # what makes decay SELECTIVE per-bucket (see decay_all()):
        # without it, a bucket that hasn't seen this board config in
        # 500 epochs would still shrink toward zero on all 500, purely
        # from existing, which defeats part of the point of splitting
        # in the first place (chat discussion, 2026-10-01). Cleared
        # every time decay_all() consumes it.
        self._touched_since_decay = set()
        # None (default) -- follow the global FULL_SPLIT_MODE flag,
        # exactly like every manager before this override existed.
        # True/False -- pin THIS manager to full-split or common-10-
        # only regardless of the global flag. Added 2026-10 after the
        # "split everything" extension's FULL_SPLIT_MODE rollout to 17
        # policies at once caused a real village-win-rate collapse
        # (confirmed via A/B test against a pre-extension backup): with
        # 1000 games/epoch spread across ~30 buckets for EACH of 17
        # policies simultaneously, none of them had remotely enough
        # samples to converge, and the half-trained buckets were net
        # harmful. The fix is this per-manager override, set via
        # force_full_split(), so village_ai.execution_split/wolf_ai.
        # attack_split (already proven, with months of training behind
        # their full split) can stay on FULL_SPLIT_MODE while any newly
        # (re-)split policy starts on the far less data-hungry common-
        # ~10-configs-only mode instead, exactly as recommended in the
        # chat discussion this shipped with.
        self.full_split_override = None

    def force_full_split(self, enabled):
        """
        Pin this manager's full-split behavior independently of the
        global FULL_SPLIT_MODE flag -- see full_split_override's
        comment in __init__ for why this exists. Call with True to
        force every distinct config into its own bucket for this
        manager regardless of the global flag; False to force the
        common-~10-configs-only grouping regardless of the global
        flag. One-way in practice (nothing stops calling it again, but
        see enable_full_split's own docstring for why widening back
        out is a one-way, data-hungry commitment once buckets exist).
        """
        self.full_split_override = enabled

    def _effective_full(self):
        return FULL_SPLIT_MODE if self.full_split_override is None else self.full_split_override

    def _new_policy(self, initial_weights=None):
        policy = policy_utils.LinearPolicy(
            len(self.base.weights),
            lr_multipliers=self.base.lr_multipliers,
            decay_multipliers=self.base.decay_multipliers,
        )
        if initial_weights is not None:
            policy.weights = list(initial_weights)
        return policy

    def enable_split(self):
        """One-way switch -- see class docstring. Idempotent."""
        self.split_enabled = True

    def active(self, day1_config):
        """The LinearPolicy that should actually be used right now."""
        if not self.split_enabled:
            return self.base
        key = bucket_key(day1_config, full=self._effective_full())
        policy = self.buckets.get(key)
        if policy is None:
            # WARM START. Ordinarily (the first time ANY bucket is
            # created) there's nothing better than the frozen shared
            # base to copy from. But under enable_full_split(), a
            # newly-split-off rare config was, until just now, having
            # its games routed into OTHER_BUCKET -- if that bucket
            # already exists, it has already accumulated some
            # experience specific to exactly this kind of rare board
            # (more relevant than the older, more generic frozen
            # base), so prefer it as the warm-start ancestor when
            # available.
            ancestor = self.buckets.get(OTHER_BUCKET, self.base)
            policy = self._new_policy(initial_weights=ancestor.weights)
            self.buckets[key] = policy
        return policy

    def record_sample(self, day1_config):
        """
        Call this once per REINFORCE update actually applied for this
        decision type (village_ai.update's "execution" branch / wolf_
        ai.update's "attack" branch each call it once) -- NOT from
        .active() itself, since .active() is also called just to make
        the decision (once per vote/attack), which would double-count
        against the once-per-update call this is meant to track.
        Before split_enabled, this is a no-op (there's only the one
        shared policy; its own sample count isn't tracked per-bucket
        because there's no bucket to attribute it to).
        """
        if not self.split_enabled:
            return
        key = bucket_key(day1_config, full=self._effective_full())
        self.sample_counts[key] = self.sample_counts.get(key, 0) + 1
        self._touched_since_decay.add(key)

    def decay_all(self, decay_rate):
        """
        Decay whichever policy/policies are currently live. Before
        the split, that's just `base` (identical to calling
        base.decay(decay_rate) directly) -- a single shared policy is
        used by essentially every game regardless of board config, so
        unconditional per-epoch decay is the right behavior there and
        is unchanged.

        After the split, decaying EVERY bucket unconditionally (the
        original behavior here) meant a config that simply hadn't
        come up in a while still shrank every epoch purely from
        existing -- a forgetting mechanism with nothing to do with
        that bucket's own training, that would keep eating into a
        rare config's learned weights even though cross-config
        gradient contamination (the other half of the forgetting
        problem) was already eliminated by the split. Fixed by only
        decaying buckets that actually received at least one
        record_sample() call since the last decay_all() -- i.e. a
        bucket that didn't train this epoch doesn't decay this epoch
        either. (See the chat discussion this shipped with, 2026-10,
        for the "block separation should mean no cross-contamination
        AND no silent erosion from disuse" reasoning.)
        """
        if not self.split_enabled:
            self.base.decay(decay_rate)
            return
        for key in self._touched_since_decay:
            policy = self.buckets.get(key)
            if policy is not None:
                policy.decay(decay_rate)
        self._touched_since_decay = set()

    # --- persistence -------------------------------------------------

    def snapshot(self):
        """For persistence.py's save_weights()."""
        return {
            "split_enabled": self.split_enabled,
            "buckets": {key: list(p.weights) for key, p in self.buckets.items()},
            "sample_counts": dict(self.sample_counts),
            "full_split_override": self.full_split_override,
        }

    def load_snapshot(self, data, pad_ok=False):
        """
        For persistence.py's load_weights(). Mirrors that file's own
        per-policy shape-mismatch handling: a bucket whose saved
        weight count doesn't match the current feature layout is
        dropped (that ONE bucket starts fresh -- lazily re-created via
        .active() the next time it's needed -- rather than discarding
        every other already-trained bucket too).
        """
        if not data:
            return
        # 2026-10-06: a (re)load replaces the buckets instead of merging
        # into whatever a previous load left in this process.
        self.buckets = {}
        self.split_enabled = data.get("split_enabled", False)
        expected_len = len(self.base.weights)
        for key, weights in data.get("buckets", {}).items():
            if pad_ok and 0 < len(weights) < expected_len:
                # features were APPENDED to this layout (2026-10-06) --
                # every old index is still valid, so pad with zeros.
                weights = list(weights) + [0.0] * (expected_len - len(weights))
            if len(weights) != expected_len:
                print(f"[board_split] saved bucket {key!r} has {len(weights)} "
                      f"entries but the current code expects {expected_len} -- "
                      "the feature layout changed since this was saved. That "
                      "one bucket starts fresh (re-warm-started from the "
                      "shared base) the next time it's used; every other "
                      "bucket still loads normally.")
                continue
            self.buckets[key] = self._new_policy(initial_weights=weights)
        # Only keep sample counts for buckets that actually loaded
        # (a dropped bucket restarts fresh, so its old count would
        # overstate how much it has trained).
        self.sample_counts = {
            k: v for k, v in dict(data.get("sample_counts", {})).items()
            if k in self.buckets
        }
        # Older saves (before this override existed) simply won't have
        # this key -- .get(...) defaults to None, meaning "follow the
        # global FULL_SPLIT_MODE flag", the exact behavior those saves
        # already had.
        self.full_split_override = data.get("full_split_override")
