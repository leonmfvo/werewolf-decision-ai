import random


def softmax(scores):
    m = max(scores)
    exps = [pow(2.718281828459045, s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


class LinearPolicy:
    """
    A minimal linear-softmax policy over a fixed-length feature vector.

        score(features) = weights . features
        action is sampled from softmax(scores) across candidates

    Weights start at all-zero, which makes every candidate score 0
    and every choice uniform random -- i.e. behaviourally IDENTICAL
    to the old `random.choice(...)` placeholder. Nothing changes
    until something actually updates `.weights`.
    """

    def __init__(self, num_features, lr_multipliers=None, decay_multipliers=None):
        """
        lr_multipliers / decay_multipliers : optional per-feature
        multiplier lists (same length as num_features). Most policies
        leave these as None (every feature learns/decays at the same
        rate). They exist for the rare case where ONE feature inside
        an otherwise-shared weight vector represents a near-certain,
        structurally-proven fact (e.g. "this candidate's claim has
        been confirmed fake by an unrelated deduction") while its
        neighbors are ordinary, noise-prone suspicion signals. A
        single global learning-rate/decay-rate can't let that one
        feature commit hard toward its deserved extreme value AND
        keep the noisy ones well-regularized at the same time -- see
        wolf_ai.py's ATTACK_LR_MULTIPLIER for the first place this
        tension was found and fixed. Defaults to 1.0 everywhere,
        which is behaviourally identical to not having this at all.
        """
        self.weights = [0.0] * num_features
        self.lr_multipliers = (
            list(lr_multipliers) if lr_multipliers is not None else [1.0] * num_features
        )
        self.decay_multipliers = (
            list(decay_multipliers) if decay_multipliers is not None else [1.0] * num_features
        )

    def score(self, features):
        return sum(w * f for w, f in zip(self.weights, features))

    def decay(self, decay_rate):
        """
        Shrink every weight slightly toward zero.

        This is how "old, stale conclusions get forgotten" actually
        works here -- there's no stored dataset of old games to
        delete, only the accumulated weight values. A belief that
        was only ever supported by ancient (now-outdated) evidence
        will fade over time unless the CURRENT policy keeps
        re-earning it through fresh wins/losses.

        Each feature decays at decay_rate * self.decay_multipliers[k],
        so a feature with a multiplier below 1.0 is allowed to hold
        onto a strong conclusion for longer than its neighbors.
        """
        self.weights = [
            w * (1.0 - decay_rate * m)
            for w, m in zip(self.weights, self.decay_multipliers)
        ]

    def choose(self, feature_list, candidates, rng=random):
        """
        feature_list : list of feature-vectors, one per candidate,
                        in the same order as `candidates`.
        candidates   : the actual candidate values to choose among
                        (player indices, strategy names, ...).

        Returns (chosen_candidate, probs) where `probs` is the full
        softmax distribution used -- kept around so a future training
        step can compute a policy-gradient update from it.
        """
        scores = [self.score(f) for f in feature_list]
        probs = softmax(scores)
        chosen = rng.choices(candidates, weights=probs, k=1)[0]
        return chosen, probs

    def update(self, feature_list, candidates, chosen, reward, learning_rate=0.01):
        """
        One REINFORCE step for a single (state, action, reward)
        sample from this policy.

            grad log pi(a|s) = features(a) - E_pi[features]
            weights += learning_rate * reward * grad

        `feature_list` / `candidates` must be the SAME candidate set
        the original decision was made over (recomputed from the
        stored state), so the gradient's baseline term is correct.
        """
        scores = [self.score(f) for f in feature_list]
        probs = softmax(scores)

        chosen_index = candidates.index(chosen)
        chosen_features = feature_list[chosen_index]

        num_features = len(self.weights)
        expected_features = [
            sum(p * f[k] for p, f in zip(probs, feature_list))
            for k in range(num_features)
        ]

        for k in range(num_features):
            gradient_k = chosen_features[k] - expected_features[k]
            self.weights[k] += (
                learning_rate * self.lr_multipliers[k] * reward * gradient_k
            )
