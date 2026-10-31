"""
Detailed policy report (English).

Unlike observe_evolution.py's short summary, this computes ACTUAL
probabilities under representative scenarios, not just a binary
"which side is bigger" label. Run standalone -- it trains fresh,
then prints the full report.

    python3 policy_report.py
"""

import policy_utils
import self_play
import wolf_ai
import village_ai
import medium_ai
import hunter_ai


NUM_EPOCHS = 150
GAMES_PER_EPOCH = 50
LEARNING_RATE = 0.02


def softmax_report(policy, feature_list, option_labels):
    scores = [policy.score(f) for f in feature_list]
    probs = policy_utils.softmax(scores)
    return list(zip(option_labels, probs))


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def report_wolf():
    section("WOLF AI")

    print(f"\n-- Day-by-day CO decision: raw weights {wolf_ai.CO_OPTIONS} --")
    w = wolf_ai.co_policy.weights
    n = wolf_ai.NUM_OPTIONS
    print("  base preference:", [round(x, 3) for x in w[:n]])
    for k, role in enumerate(wolf_ai.CLAIMABLE_ROLES):
        bonus = [w[n + j * wolf_ai.NUM_CONTEXT_FEATURES + k] for j in range(n)]
        print(f"  {role}_already_revealed bonus:", [round(x, 3) for x in bonus])

    print("\n-- CO decision: concrete scenarios (day 2) --")

    def co_state(medium_revealed, hunter_revealed):
        co_history = [None] * 7
        if medium_revealed:
            co_history[2] = "Ps"
        if hunter_revealed:
            co_history[3] = "Hn"
        return {"day": 2, "co_history": co_history}

    scenarios = {
        "nothing revealed yet": co_state(False, False),
        "Medium already revealed": co_state(True, False),
        "Hunter already revealed": co_state(False, True),
        "both already revealed": co_state(True, True),
    }
    for label, state in scenarios.items():
        feature_list = wolf_ai._co_feature_list(state)
        probs = softmax_report(wolf_ai.co_policy, feature_list, wolf_ai.CO_OPTIONS)
        formatted = ", ".join(f"{opt}={p:.3f}" for opt, p in probs)
        print(f"  {label:<28}: {formatted}")

    print("\n-- Night attack target: raw feature weights --")
    w = wolf_ai.attack_policy.weights
    for label, weight in zip(
        ["bias", "is_medium_claimant", "is_known_real_medium", "is_known_hunter", "day"], w
    ):
        print(f"  {label:<24}: {weight:+.3f}")

    print("\n-- Night attack target: concrete scenario (day 2) --")
    print("   Choosing among: a lone Medium claimant (genuine), a Hunter claimant, "
          "and a silent player.")

    def feat(is_ps, is_real, is_hunter, day):
        return [1.0, is_ps, is_real, is_hunter, day / 7.0]

    day = 2
    candidates = ["medium claimant", "hunter claimant", "silent player"]
    feature_list = [
        feat(1.0, 1.0, 0.0, day),
        feat(0.0, 0.0, 1.0, day),
        feat(0.0, 0.0, 0.0, day),
    ]
    for label, p in softmax_report(wolf_ai.attack_policy, feature_list, candidates):
        print(f"  P(attack {label:<18}) = {p:.4f}")


def report_village():
    section("VILLAGE AI (lookup table)")

    print("\n-- Table cell scores for representative situations (day 2, no guard success yet) --")
    scenarios = {
        "sole Medium claimant (uncontested)": ("Ps", 1, 0),
        "one of TWO contradicting Medium claims": ("Ps", 2, 0),
        "sole Hunter claimant (uncontested)": ("Hn", 0, 1),
        "one of TWO contradicting Hunter claims": ("Hn", 0, 2),
        "silent, unclaimed player": (None, 1, 1),
    }
    raw_scores = {}
    for label, (own_co, mc, hc) in scenarios.items():
        raw_scores[label] = village_ai.lookup(alive_count=5, own_co=own_co, medium_count=mc, hunter_count=hc)
        print(f"  {label:<40}: raw score = {raw_scores[label]:+.3f}")

    print("\n-- Same scenarios, as an actual probability (softmax against the others) --")
    probs = policy_utils.softmax(list(raw_scores.values()))
    for label, p in zip(scenarios.keys(), probs):
        print(f"  P(execute | {label:<40}) = {p:.4f}")

    print("\n-- Effect of guard-success COUNT so far, on a sole Hunter claim --")
    for gc in range(3):
        s = village_ai.lookup(alive_count=5, own_co="Hn", medium_count=0, hunter_count=1, guard_count=gc)
        print(f"  score, {gc} guard success(es) so far: {s:+.3f}")

    print("\n-- Effect of being a CONFIRMED villager (via a revealed guard save) --")
    not_confirmed = village_ai.lookup(alive_count=5, own_co=None, medium_count=1, hunter_count=1, is_confirmed_villager=False)
    confirmed = village_ai.lookup(alive_count=5, own_co=None, medium_count=1, hunter_count=1, is_confirmed_villager=True)
    print(f"  score, silent + NOT confirmed: {not_confirmed:+.3f}")
    print(f"  score, silent + confirmed villager (guard saved them): {confirmed:+.3f}")

    print("\n-- Effect of claiming plain Villager (experimental -- normally banned in human play) --")
    silent_score = village_ai.lookup(alive_count=5, own_co=None, medium_count=1, hunter_count=1)
    vi_claim_score = village_ai.lookup(alive_count=5, own_co="Vi", medium_count=1, hunter_count=1)
    print(f"  score, stayed silent      : {silent_score:+.3f}")
    print(f"  score, claimed 'Villager' : {vi_claim_score:+.3f}")


def report_medium():
    section("MEDIUM AI")

    print("\n-- P(reveal), still silent, by day --")
    for day in [1, 2, 3, 4]:
        feature_list = medium_ai._co_feature_list({"day": day})
        for label, p in softmax_report(medium_ai.co_policy, feature_list,
                                        medium_ai.CO_OPTIONS):
            if label == "reveal":
                print(f"  day {day}: P(reveal) = {p:.4f}")


def report_hunter():
    section("HUNTER AI")

    print("\n-- P(reveal), still silent, by guard-success-count / gray-count "
          "(day no longer matters -- see note below) --")
    for gray in range(6):
        for gc in range(3):
            state = {"guard_success_count": gc, "gray_count": gray}
            feature_list = hunter_ai._co_feature_list(state)
            for label, p in softmax_report(hunter_ai.co_policy, feature_list,
                                            hunter_ai.CO_OPTIONS):
                if label == "reveal":
                    print(f"  gray={gray}, guards={gc}: P(reveal) = {p:.4f}")

    print("\n-- Guard target: raw feature weights --")
    for label, weight in zip(
        ["bias", "is_medium_claimant", "day"], hunter_ai.guard_policy.weights
    ):
        print(f"  {label:<24}: {weight:+.3f}")


def full_report():
    report_wolf()
    report_village()
    report_medium()
    report_hunter()
    print()


def main():
    for epoch in range(1, NUM_EPOCHS + 1):
        results = self_play.run_self_play(GAMES_PER_EPOCH, verbose=False)
        self_play.update_policies(results, learning_rate=LEARNING_RATE)

    full_report()


if __name__ == "__main__":
    main()
