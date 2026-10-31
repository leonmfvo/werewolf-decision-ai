"""
Automated strategy search.

Instead of manually hypothesizing one fixed strategy at a time and
hand-deriving its win rate, this defines a PARAMETERIZED family of
strategies and evaluates all of them automatically, ranking the
results so the best (and worst) candidates can be identified without
a human proposing each one individually.

This is deliberately separate from the self-play learning system
(main.py / self_play.py etc.) -- it evaluates fixed, hand-specified
policies via Monte Carlo, which is far more reliable for testing a
specific "what if everyone did exactly X" hypothesis than waiting
for decentralized RL to discover it on its own.
"""

import random
from itertools import product


ROLES = ["Wo", "Wo", "Ps", "Hn", "Vi", "Vi", "Vi"]
WOLVES = [0, 1]
MEDIUM = 2
HUNTER = 3


def simulate_one_game(medium_reveal_day, hunter_reveal_day, hunter_guards_medium):
    """
    Play one game under a FIXED (non-learned) strategy:

    - Medium reveals on `medium_reveal_day` (if still alive by then).
    - Hunter reveals on `hunter_reveal_day` (if still alive by then).
    - Every night, while alive, the Hunter guards the Medium if
      `hunter_guards_medium` is True AND the Medium is alive
      (regardless of whether the Hunter has revealed yet -- guarding
      is always a private action). Otherwise guards an arbitrary
      other player (no meaningful protection).
    - Village always executes uniformly at random among the living,
      NOT-YET-REVEALED players (this is the one part carried over
      from what self-play consistently learned: never execute a
      revealed Medium/Hunter).
    - Wolves attack the highest-priority REVEALED target if one
      exists (Hunter over Medium, matching the learned attack
      weights), otherwise a random living non-wolf.
    """

    alive = [True] * 7
    revealed = [False] * 7
    day = 1

    while True:
        wolves_alive = sum(alive[i] for i in WOLVES)
        village_alive = sum(alive) - wolves_alive
        if wolves_alive == 0:
            return "Village"
        if wolves_alive >= village_alive:
            return "Werewolf"

        # Reveal phase
        if day == medium_reveal_day and alive[MEDIUM] and not revealed[MEDIUM]:
            revealed[MEDIUM] = True
        if day == hunter_reveal_day and alive[HUNTER] and not revealed[HUNTER]:
            revealed[HUNTER] = True

        # Day execution: random among living, unrevealed players
        gray = [i for i in range(7) if alive[i] and not revealed[i]]
        if gray:
            alive[random.choice(gray)] = False

        wolves_alive = sum(alive[i] for i in WOLVES)
        village_alive = sum(alive) - wolves_alive
        if wolves_alive == 0:
            return "Village"
        if wolves_alive >= village_alive:
            return "Werewolf"

        # Night: Hunter chooses a guard target (private action, works
        # even before Hunter has revealed)
        guard_target = None
        if alive[HUNTER]:
            if hunter_guards_medium and alive[MEDIUM]:
                guard_target = MEDIUM
            else:
                others = [i for i in range(7) if alive[i] and i != HUNTER]
                if others:
                    guard_target = random.choice(others)

        # Wolves attack: prefer a REVEALED Hunter, then a REVEALED
        # Medium, else random among living non-wolves.
        if revealed[HUNTER] and alive[HUNTER]:
            attack_target = HUNTER
        elif revealed[MEDIUM] and alive[MEDIUM]:
            attack_target = MEDIUM
        else:
            non_wolf = [i for i in range(7) if alive[i] and i not in WOLVES]
            attack_target = random.choice(non_wolf) if non_wolf else None

        if attack_target is not None and attack_target != guard_target:
            alive[attack_target] = False

        day += 1


def evaluate_strategy(medium_reveal_day, hunter_reveal_day, hunter_guards_medium, trials=30000):
    wins = sum(
        1 for _ in range(trials)
        if simulate_one_game(medium_reveal_day, hunter_reveal_day, hunter_guards_medium) == "Village"
    )
    return wins / trials


def search(days_to_try=(1, 2, 3), trials=30000):
    """
    Exhaustively evaluate every combination in the parameter grid
    and return a list of (win_rate, params), sorted best-first.
    """
    results = []
    for medium_day, hunter_day, guards in product(days_to_try, days_to_try, [False, True]):
        rate = evaluate_strategy(medium_day, hunter_day, guards, trials=trials)
        results.append((rate, medium_day, hunter_day, guards))

    results.sort(reverse=True)
    return results


if __name__ == "__main__":
    print("Searching over: medium_reveal_day x hunter_reveal_day x hunter_guards_medium")
    print(f"{'village win%':>13} {'medium day':>11} {'hunter day':>11} {'guards medium':>14}")

    results = search()
    for rate, m_day, h_day, guards in results:
        print(f"{rate*100:12.2f}% {m_day:11d} {h_day:11d} {str(guards):>14}")

    best = results[0]
    print()
    print(f"BEST FOUND: medium_reveal_day={best[1]}, hunter_reveal_day={best[2]}, "
          f"hunter_guards_medium={best[3]}  ->  {best[0]*100:.2f}%")
