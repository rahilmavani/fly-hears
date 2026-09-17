"""Shared feature extraction for training and the live demo."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from scripts import encode, fly_lif

BINS = 4
SEED = 0


@dataclass
class Graph:
    weights: sp.csr_matrix
    jo_a: np.ndarray
    jo_b: np.ndarray
    jo_other: np.ndarray
    hop: np.ndarray
    types: np.ndarray

    @property
    def size(self):
        return self.weights.shape[0]


def load_graph(path: Path):
    with np.load(path, allow_pickle=False) as data:
        size = int(data['n'])
        weights = sp.csr_matrix(
            (data['W_data'], data['W_indices'], data['W_indptr']),
            shape=(size, size),
        )
        return Graph(weights, data['jo_a'], data['jo_b'], data['jo_other'],
                     data['hop'], data['types'])


def simulate_batch(graph, energies, record=False):
    """Return bin-major spike counts; optionally retain the first clip's spike times."""
    batch_size = len(energies)
    probabilities = []
    for energy in energies:
        input_indices, drive = encode.drive_matrix(
            energy, graph.jo_a, graph.jo_b, graph.jo_other,
        )
        probabilities.append(drive)
    probabilities = np.stack(probabilities, axis=-1)
    binned = np.zeros((BINS, graph.size, batch_size), dtype=np.int32)
    frames = []
    bin_length = encode.STEPS // BINS

    def collect(step, spikes):
        binned[min(step // bin_length, BINS - 1)] += spikes
        if record:
            frames.append(np.flatnonzero(spikes[:, 0]))

    counts = fly_lif.run_lif(
        graph.weights,
        lambda step: (input_indices, probabilities[:, step, :]),
        encode.STEPS,
        batch_size,
        rng=np.random.default_rng(SEED),
        record=collect,
    )
    features = binned.transpose(2, 0, 1).reshape(batch_size, BINS * graph.size)
    return features, counts, frames


def audio_features(energies):
    return energies.reshape(len(energies), encode.N_BANDS, BINS, -1).mean(axis=-1).reshape(len(energies), -1)
