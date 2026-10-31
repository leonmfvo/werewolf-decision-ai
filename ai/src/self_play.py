import main
import wolf_ai
import village_ai
import medium_ai
import hunter_ai
import seer_ai
import villager_ai
import madman_ai
import value_function


# side name (as logged in history) -> (winner label that means "this
# side won", the module holding that side's policies)
SIDE_INFO = {
    "wolf": ("Werewolf", wolf_ai),
    "village": ("Village", village_ai),
    "medium": ("Village", medium_ai),
    "hunter": ("Village", hunter_ai),
    "seer": ("Village", seer_ai),
    "villager": ("Village", villager_ai),
    # The Madman is counted as VILLAGE for the numeric win-condition
    # (config.py's comment on "Kg" explains why), but WINS alongside
    # the werewolves -- so its winner label is "Werewolf" even though
    # its every observable action (a vote, in this step) looks
    # exactly like something a villager could have done.
    "madman": ("Werewolf", madman_ai),
}


# ============================================================
# Self-play
# ============================================================

def run_self_play(num_games, verbose=False):
    """
    Play `num_games` games using the CURRENT wolf_ai / village_ai
    policies, collecting each game's outcome and full decision
    trajectory.

    Returns
    -------
    list[dict]
        One entry per game, each shaped like main.run_game()'s
        return value: {"winner": ..., "history": [...], "roles": [...]}
    """
    return [main.run_game(verbose=verbose) for _ in range(num_games)]


# ============================================================
# Result summary
# ============================================================

def summarize(results):

    total = len(results)
    wolf_wins = sum(1 for r in results if r["winner"] == "Werewolf")
    village_wins = total - wolf_wins

    print(f"Games played  : {total}")
    print(f"Werewolf wins : {wolf_wins} ({wolf_wins / total:.1%})")
    print(f"Village wins  : {village_wins} ({village_wins / total:.1%})")

    return {
        "total": total,
        "wolf_win_rate": wolf_wins / total,
        "village_win_rate": village_wins / total,
    }


# ============================================================
# Policy update
# ============================================================

# Running-average reward baseline, tracked per (side, decision type).
# e.g. baselines[("village", "execution")] ~= village's average
# reward from execution decisions so far. Subtracting this from the
# raw +1/-1 reward is "REINFORCE with baseline": it doesn't change
# which direction the gradient points (the expected update is the
# same), but it cuts the variance a lot -- a decision in a game
# that was lost anyway (as usual) is no longer blamed as hard as a
# decision in one of the rarer wins is credited.
_baselines = {}
BASELINE_ALPHA = 0.02  # how fast the running average adapts


def _get_baseline(key):
    return _baselines.get(key, 0.0)


def _update_baseline(key, reward):
    old = _baselines.get(key, 0.0)
    _baselines[key] = old + BASELINE_ALPHA * (reward - old)


def decay_all_policies(decay_rate=0.001, freeze_sides=()):
    """
    Shrink every AI module's weights slightly toward zero.

    Call this once per epoch (after update_policies) to let old,
    no-longer-supported conclusions fade over time, instead of
    staying frozen forever once a policy becomes very confident.

    Parameters
    ----------
    freeze_sides : iterable of str
        Side names (e.g. "wolf") to SKIP entirely -- their weights
        are left completely untouched (no decay either), so a
        deliberately fixed opponent stays exactly fixed.
    """
    # 2026-10 "split everything" extension: every policy that used to
    # call plain_policy.decay(...) directly here now goes through its
    # BoardSplitManager's .decay_all(...) instead (see board_split.py
    # and the attack_split/execution_split precedent this follows) --
    # this decays whichever policy/policies are actually live right
    # now (the shared base before enable_board_split(), only the
    # buckets actually touched since the last call once split, per the
    # 2026-10-01 selective-decay fix) rather than the raw shared
    # object directly. Before enable_split() is ever called on a given
    # manager, .decay_all() is IDENTICAL to calling .decay() on the
    # wrapped policy directly -- zero behavior change for any save
    # file/run that never opts into splitting.
    #
    # Policies that were deliberately NEVER decayed before this
    # extension (wolf_ai's fake_declaration/fake_report/fake_seer_
    # report/accuse_target, and madman_ai's fake_report/fake_seer_
    # report/accuse_target) are still wrapped in their own
    # BoardSplitManager (for the split/no-cross-contamination benefit
    # to choose()/update()), but are deliberately left OUT of this
    # function -- same "no decay for this one" precedent as before,
    # just applied consistently to each one's split manager instead of
    # its raw policy object.
    #
    # wolf_ai.co_policy / hunter_ai.co_policy / madman_ai.co_policy
    # (and wolf_ai.day1_joint_co_policy, already excluded before this
    # extension) are NOT wrapped in a BoardSplitManager at all (2026-10
    # user decision -- see board_split.py's module docstring): a
    # reveal/stay-silent timing decision is part of what PRODUCES the
    # board configuration, so splitting it BY that configuration is
    # backwards. These three keep calling plain_policy.decay(...)
    # directly, exactly as before this extension.
    if "wolf" not in freeze_sides:
        wolf_ai.co_policy.decay(decay_rate)
        # attack_policy uses a lower decay than the rest of the wolf
        # side -- see the comment by ATTACK_DECAY_MULTIPLIER in
        # wolf_ai.py for why "who to bite" needs to be allowed to
        # commit harder than "who to vote for" does.
        wolf_ai.attack_split.decay_all(decay_rate * wolf_ai.ATTACK_DECAY_MULTIPLIER)
        wolf_ai.vote_split.decay_all(decay_rate)
        wolf_ai.endgame_declare_split.decay_all(decay_rate)
        wolf_ai.day1_joint_co_policy.decay(decay_rate)
    if "village" not in freeze_sides:
        village_ai.execution_split.decay_all(decay_rate)
        village_ai.guard_instruction_split.decay_all(decay_rate)
        village_ai.guard_instruction_split_1.decay_all(decay_rate)
        village_ai.guard_instruction_split_3.decay_all(decay_rate)
    if "medium" not in freeze_sides:
        medium_ai.nomination_split.decay_all(decay_rate)
        medium_ai.nomination_target_split.decay_all(decay_rate)
    if "hunter" not in freeze_sides:
        hunter_ai.co_policy.decay(decay_rate)
        hunter_ai.guard_split.decay_all(decay_rate)
    if "seer" not in freeze_sides:
        seer_ai.investigate_split.decay_all(decay_rate)
    if "villager" not in freeze_sides:
        villager_ai.endgame_bluff_split.decay_all(decay_rate)
    if "madman" not in freeze_sides:
        madman_ai.vote_split.decay_all(decay_rate)
        madman_ai.co_policy.decay(decay_rate)
        madman_ai.endgame_reveal_split.decay_all(decay_rate)
        # fake_report_split / fake_seer_report_split / accuse_target_split
        # are deliberately NOT decayed here -- see this function's own
        # docstring comment above.


def update_policies(results, learning_rate=0.01, use_baseline=True, freeze_sides=(),
                     use_value_function=False, value_learning_rate=0.02):
    """
    Turn a batch of self-play results into REINFORCE updates.

    Every decision made by the winning side in a given game gets
    reward +1; every decision made by the losing side gets -1. This
    is still full-Monte-Carlo credit assignment (the whole game's
    outcome is applied to every decision in it, uniformly) -- a
    single Day-1 CO gets blamed/credited exactly as much as a
    Day-3 attack, regardless of which one actually mattered.

    Two ways to reduce the resulting variance are available:

    - use_baseline (default): subtract a running-average baseline
      per (side, decision type). Lightweight, but the baseline is a
      CONSTANT -- it doesn't know whether THIS SPECIFIC state was
      already a likely win or a likely loss.

    - use_value_function: use value_function.py's learned Critic
      instead. It predicts an expected outcome FOR THIS EXACT
      STATE, so a good decision made in an already-bad position can
      still get credit for being less bad than expected (and vice
      versa) -- a real fix for the "every decision in a game blamed
      equally" problem, not just a lighter mitigation. The value
      function itself is trained on every state seen, from every
      side, in the same pass. If both use_baseline and
      use_value_function are True, the value function takes
      priority (baseline is skipped).

    Parameters
    ----------
    freeze_sides : iterable of str
        Side names (e.g. "wolf") whose decisions are logged for
        the game but get NO gradient update -- useful for training
        one side against a deliberately fixed opponent, instead of
        full co-evolution. Pass () (default) for normal co-evolution.
    """

    for result in results:

        winner = result["winner"]
        village_outcome = 1.0 if winner == "Village" else -1.0

        for record in result["history"]:

            side = record["side"]

            # 2026-10-06: the critic's prediction is taken BEFORE it is
            # fitted to this very game's outcome. Predicting after the
            # update leaked the outcome into the baseline (it shrank the
            # advantage toward 0 for the states just fitted).
            predicted_before = (
                value_function.predict(record["state"], side) if use_value_function else None
            )
            if use_value_function:
                value_function.update(record["state"], village_outcome, value_learning_rate)

            if side in freeze_sides:
                continue

            winner_label, module = SIDE_INFO[side]

            reward = 1.0 if winner == winner_label else -1.0

            if use_value_function:
                signal = reward - predicted_before
            elif use_baseline:
                key = (side, record["type"])
                baseline = _get_baseline(key)
                signal = reward - baseline
                _update_baseline(key, reward)
            else:
                signal = reward

            module.update(
                record["type"],
                record["state"],
                record["action"],
                signal,
                learning_rate,
            )


# ============================================================
# Training loop
# ============================================================

def train(num_epochs, games_per_epoch=50, learning_rate=0.02, verbose=False):
    """
    games_per_epoch defaults to a SMALL batch (50), not a large one.

    This matters more than it looks: a large batch makes each
    policy update very confident (low variance), which lets a side
    lock onto a strategy that was only correct against the
    OPPONENT'S OLD policy. By the time the opponent adapts, the
    confident side's gradient has already vanished and it can't
    correct. Smaller batches make each update noisier but keep
    both sides able to keep tracking each other as they co-adapt.
    """

    for epoch in range(1, num_epochs + 1):

        results = run_self_play(games_per_epoch, verbose=verbose)

        print(f"--- Epoch {epoch} ---")
        summarize(results)

        update_policies(results, learning_rate=learning_rate)

        print()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    train(num_epochs=100)
