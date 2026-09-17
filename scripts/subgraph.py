"""Extract up to 30,000 neurons within three connections of Johnston's-organ inputs."""

import time

import numpy as np
import scipy.sparse as sp

from scripts import fly_lif

HOPS = 3
MAX_NEURONS = 30_000


def main():
    started = time.monotonic()
    neurons = fly_lif.load_neurons()
    sources, targets, counts = fly_lif.load_edges(neurons)
    types = neurons.type.fillna('')
    sensory = np.flatnonzero(types.str.match(r'^JO-').to_numpy())
    jo_a = np.flatnonzero(types.str.match(r'^JO-A').to_numpy())
    jo_b = np.flatnonzero(types.str.match(r'^JO-B').to_numpy())
    jo_other = np.setdiff1d(sensory, np.concatenate([jo_a, jo_b]))
    if not len(sensory) or len(sensory) > MAX_NEURONS:
        raise ValueError('The sensory-neuron count is incompatible with the subgraph limit.')

    outgoing = sp.csr_matrix((counts, (sources, targets)), shape=(len(neurons), len(neurons)))
    selected = np.zeros(len(neurons), dtype=bool)
    selected[sensory] = True
    distance = np.full(len(neurons), -1, dtype=np.int32)
    distance[sensory] = 0
    frontier = sensory
    for hop in range(1, HOPS + 1):
        strength = np.asarray(outgoing[frontier].sum(axis=0)).ravel()
        candidates = np.flatnonzero((strength > 0) & ~selected)
        remaining = MAX_NEURONS - int(selected.sum())
        order = np.argsort(-strength[candidates], kind='stable')[:remaining]
        frontier = candidates[order]
        selected[frontier] = True
        distance[frontier] = hop
        print(f'Hop {hop}: added {len(frontier):,}; total {selected.sum():,}', flush=True)
        if selected.sum() >= MAX_NEURONS or not len(frontier):
            break

    indices = np.flatnonzero(selected)
    local = np.full(len(neurons), -1, dtype=np.int32)
    local[indices] = np.arange(len(indices))
    keep = selected[sources] & selected[targets]
    full_weights = fly_lif.signed_matrix(neurons, sources[keep], targets[keep], counts[keep])
    weights = full_weights[indices][:, indices].tocsr()
    output = fly_lif.DATA / 'auditory.npz'
    np.savez_compressed(
        output, W_data=weights.data, W_indices=weights.indices, W_indptr=weights.indptr,
        n=len(indices), global_idx=indices, body_ids=neurons.bodyId.to_numpy()[indices],
        jo_a=local[jo_a], jo_b=local[jo_b], jo_other=local[jo_other],
        hop=distance[indices], types=types.to_numpy()[indices].astype(str),
    )
    print(f'Saved {output}: {len(indices):,} neurons, {weights.nnz:,} signed connections')
    print(f'Finished in {time.monotonic() - started:.0f} seconds')


if __name__ == '__main__':
    main()
