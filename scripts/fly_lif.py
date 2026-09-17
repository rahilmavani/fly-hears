"""Frozen connectome weights and a simplified leaky integrate-and-fire simulator."""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

DT = 1.0
TAU_M = 20.0
TAU_SYN = 5.0
V_REST = -52.0
V_THR = -45.0
V_RESET = -52.0
REFRAC_STEPS = 2
DELAY = 2
W_UNIT = 0.275
WEIGHT_MIN = 5
NT_SIGN = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1, 'histamine': -1}


def load_neurons():
    annotations = pd.read_feather(DATA / 'annotations.feather')
    neurons = annotations[annotations.status == 'Traced'].reset_index(drop=True)
    transmitters = pd.read_feather(DATA / 'neurotransmitters.feather').set_index('body')
    neurons['nt'] = neurons.bodyId.map(transmitters.consensus_nt).fillna('unknown')
    return neurons


def load_edges(neurons):
    index = pd.Series(np.arange(len(neurons)), index=neurons.bodyId.values)
    edges = pd.read_feather(DATA / 'weights.feather')
    edges = edges[edges.weight >= WEIGHT_MIN]
    edges = edges[edges.body_pre.isin(index.index) & edges.body_post.isin(index.index)]
    sources = index[edges.body_pre.values].to_numpy()
    targets = index[edges.body_post.values].to_numpy()
    counts = edges.weight.to_numpy(np.float32)
    kenyon = (neurons['class'] == 'Kenyon_Cell').to_numpy()
    keep = ~(kenyon[sources] & kenyon[targets])
    return sources[keep], targets[keep], counts[keep]


def signed_matrix(neurons, sources, targets, counts):
    """Rows are receiving neurons; columns are sending neurons."""
    signs = neurons.nt.map(NT_SIGN).fillna(0).to_numpy(np.float32)
    weights = counts * W_UNIT * signs[sources]
    keep = weights != 0
    return sp.csr_matrix(
        (weights[keep], (targets[keep], sources[keep])),
        shape=(len(neurons), len(neurons)), dtype=np.float32,
    )


def run_lif(weights, drive, steps, batch, rng=None, record=None):
    rng = np.random.default_rng(0) if rng is None else rng
    size = weights.shape[0]
    membrane_decay = np.exp(-DT / TAU_M)
    synapse_decay = np.exp(-DT / TAU_SYN)
    voltage = np.full((size, batch), V_REST, dtype=np.float32)
    current = np.zeros((size, batch), dtype=np.float32)
    refractory = np.zeros((size, batch), dtype=np.int32)
    delayed = [np.zeros((size, batch), dtype=np.float32) for _ in range(DELAY + 1)]
    counts = np.zeros((size, batch), dtype=np.int32)

    for step in range(steps):
        incoming = delayed[step % len(delayed)]
        current = current * synapse_decay + incoming
        incoming[:] = 0
        voltage = V_REST + (voltage - V_REST) * membrane_decay + current * (1 - membrane_decay)
        voltage[refractory > 0] = V_RESET
        spikes = voltage >= V_THR

        if drive is not None:
            indices, probabilities = drive(step)
            if indices is not None and len(indices):
                # Identical random draws make each clip independent of batch size and ordering.
                draws = rng.random((len(indices), 1), dtype=np.float32)
                spikes[indices] |= draws < probabilities
        spikes &= refractory <= 0
        voltage[spikes] = V_RESET
        current[spikes] = 0
        refractory[spikes] = REFRAC_STEPS
        refractory -= 1
        if spikes.any():
            delayed[(step + DELAY) % len(delayed)] += weights @ spikes.astype(np.float32)
        counts += spikes
        if record is not None:
            record(step, spikes)
    return counts
