"""
Curriculum learning + league play.

Two techniques from the "how to make co-evolution converge more
reliably" discussion, built on top of self_play.py's existing
freeze_sides mechanism:

  - CURRICULUM LEARNING: train one side against a FROZEN opponent
    first (so it can build real competence against a stationary
    target, the way "frozen wolf -> village reached 94.7%" showed
    earlier), THEN unfreeze and let full co-evolution continue from
    that stronger starting point -- instead of both sides starting
    from all-zero weights and chasing a constantly-moving target
    from turn one.

  - LEAGUE PLAY: keep a pool of past snapshots of each side's
    weights, and periodically have the OTHER sides play (and learn
    from) games against a RANDOMLY SAMPLED PAST version, not just
    whatever the live version happens to be doing right now. This
    is the same idea AlphaStar and similar self-play systems use to
    stop training from overfitting to the latest opponent and
    oscillating (like the villager-bluffing case that never settled
    even after 800 epochs).
"""

import random

import self_play
import wolf_ai
import village_ai
import medium_ai
import hunter_ai
import villager_ai


# ============================================================
# Snapshotting: capture / restore all policy weights for one side
# ============================================================

SIDE_MODULES = {
    "wolf": [
        ("co_policy", wolf_ai),
        ("attack_policy", wolf_ai),
        ("vote_policy", wolf_ai),
        ("fake_declaration_policy", wolf_ai),
        ("fake_report_policy", wolf_ai),
    ],
    "village": [
        ("execution_policy", village_ai),
        ("guard_instruction_policy", village_ai),
    ],
    "medium": [
        ("co_policy", medium_ai),
        ("nomination_policy", medium_ai),
        ("nomination_target_policy", medium_ai),
    ],
    "hunter": [
        ("co_policy", hunter_ai),
        ("guard_policy", hunter_ai),
    ],
    "villager": [
        ("co_policy", villager_ai),
    ],
}


def get_side_weights(side):
    """Snapshot every policy's weights belonging to `side` right now."""
    return {
        attr: list(getattr(module, attr).weights)
        for attr, module in SIDE_MODULES[side]
    }


def set_side_weights(side, snapshot):
    """Overwrite `side`'s LIVE policy weights with a snapshot."""
    for attr, module in SIDE_MODULES[side]:
        getattr(module, attr).weights = list(snapshot[attr])


# ============================================================
# League pool: a history of past snapshots per side
# ============================================================

league_pool = {side: [] for side in SIDE_MODULES}


def save_to_pool(side, max_pool_size=20):
    """Add the CURRENT live weights for `side` to its league pool."""
    league_pool[side].append(get_side_weights(side))
    if len(league_pool[side]) > max_pool_size:
        league_pool[side].pop(0)


def sample_from_pool(side):
    """A random past snapshot for `side`, or None if the pool is empty."""
    if not league_pool[side]:
        return None
    return random.choice(league_pool[side])


def run_league_epoch(games_per_epoch=50, learning_rate=0.03, decay_rate=0.03,
                      league_side="wolf", league_fraction=0.3, use_value_function=False):
    """
    One epoch, split into two batches:
      - a LIVE batch (both sides use current weights, both learn)
      - a LEAGUE batch (`league_side` is temporarily swapped to a
        random past snapshot; only the OTHER sides learn from these
        games, since `league_side`'s own live policy didn't
        actually make these decisions)
    """

    league_games = int(games_per_epoch * league_fraction)
    live_games = games_per_epoch - league_games

    if live_games > 0:
        results = self_play.run_self_play(live_games, verbose=False)
        self_play.update_policies(results, learning_rate=learning_rate,
                                   use_value_function=use_value_function)

    if league_games > 0:
        past = sample_from_pool(league_side)
        if past is not None:
            live_snapshot = get_side_weights(league_side)
            set_side_weights(league_side, past)
            results = self_play.run_self_play(league_games, verbose=False)
            set_side_weights(league_side, live_snapshot)
            self_play.update_policies(results, learning_rate=learning_rate,
                                       freeze_sides=(league_side,),
                                       use_value_function=use_value_function)

    self_play.decay_all_policies(decay_rate=decay_rate)


# ============================================================
# Curriculum training: staged freeze -> unfreeze -> (optional) league
# ============================================================

def run_curriculum_training(phase1_epochs=300, phase2_epochs=300,
                             games_per_epoch=50, learning_rate=0.03, decay_rate=0.03,
                             freeze_side="wolf", use_league_in_phase2=True,
                             league_fraction=0.3, snapshot_every=25,
                             use_value_function=False, verbose=True):
    """
    Stage 1: `freeze_side` is frozen; every OTHER side trains
    against that stationary target, so they reach real competence
    before co-evolution begins (matching the earlier "frozen wolf ->
    94.7% village" result).

    Stage 2: everyone unfreezes and co-evolution resumes -- from a
    much stronger starting point than all-zero weights. If
    use_league_in_phase2, this also periodically snapshots each
    side into the league pool and mixes in league-play batches, to
    reduce the "latest-opponent-only" oscillation risk.
    """

    if verbose:
        print(f"=== 第1段階: {freeze_side}固定・他陣営のみ学習 "
              f"({phase1_epochs}エポック) ===")

    for epoch in range(phase1_epochs):
        results = self_play.run_self_play(games_per_epoch, verbose=False)
        self_play.update_policies(results, learning_rate=learning_rate,
                                   freeze_sides=(freeze_side,),
                                   use_value_function=use_value_function)
        self_play.decay_all_policies(decay_rate=decay_rate, freeze_sides=(freeze_side,))

    if verbose:
        eval_results = self_play.run_self_play(500, verbose=False)
        wins = sum(1 for r in eval_results if r["winner"] == "Village")
        print(f"第1段階終了時点の村勝率: {wins / 500 * 100:.1f}%")
        print(f"=== 第2段階: 解凍して共進化 "
              f"({phase2_epochs}エポック{'、リーグ戦あり' if use_league_in_phase2 else ''}) ===")

    for side in SIDE_MODULES:
        save_to_pool(side)  # seed the pool with the phase-1 result

    for epoch in range(phase2_epochs):

        if use_league_in_phase2:
            run_league_epoch(games_per_epoch, learning_rate, decay_rate,
                              league_side=freeze_side, league_fraction=league_fraction,
                              use_value_function=use_value_function)
        else:
            results = self_play.run_self_play(games_per_epoch, verbose=False)
            self_play.update_policies(results, learning_rate=learning_rate,
                                       use_value_function=use_value_function)
            self_play.decay_all_policies(decay_rate=decay_rate)

        if use_league_in_phase2 and (epoch + 1) % snapshot_every == 0:
            for side in SIDE_MODULES:
                save_to_pool(side)

    if verbose:
        eval_results = self_play.run_self_play(500, verbose=False)
        wins = sum(1 for r in eval_results if r["winner"] == "Village")
        print(f"第2段階終了時点の村勝率: {wins / 500 * 100:.1f}%")
