"""A scrambled graph that retains degrees and each sender's signed weights."""

import numpy as np
import scipy.sparse as sp


def rewire(weights, seed=0, attempts_per_edge=5):
    edges = weights.tocoo()
    sources = edges.col.copy()
    targets = edges.row.copy()
    occupied = set(zip(targets.tolist(), sources.tolist()))
    rng = np.random.default_rng(seed)
    accepted = 0

    for sign in (1, -1):
        indices = np.flatnonzero(np.sign(edges.data) == sign)
        if len(indices) < 2:
            continue
        attempts = attempts_per_edge * len(indices)
        for start in range(0, attempts, 100_000):
            pairs = rng.choice(indices, size=(min(100_000, attempts - start), 2))
            for first, second in pairs:
                source_a, source_b = int(sources[first]), int(sources[second])
                target_a, target_b = int(targets[first]), int(targets[second])
                if source_a == source_b or target_a == target_b:
                    continue
                if source_a == target_b or source_b == target_a:
                    continue
                new_a, new_b = (target_b, source_a), (target_a, source_b)
                if new_a in occupied or new_b in occupied:
                    continue
                occupied.remove((target_a, source_a))
                occupied.remove((target_b, source_b))
                occupied.update((new_a, new_b))
                targets[first], targets[second] = target_b, target_a
                accepted += 1

    scrambled = sp.csr_matrix((edges.data, (targets, sources)), shape=weights.shape)
    return scrambled, {'seed': seed, 'attempts_per_edge': attempts_per_edge, 'accepted_swaps': accepted}
