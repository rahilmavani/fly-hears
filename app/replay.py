"""A compact, faithful view of a prediction's recorded spikes and connectome edges."""
import numpy as np


STAGE_LABELS = ("Sensory input", "First connections", "Deeper network")
STAGE_LIMITS = (32, 48, 64)


def build_replay(W, hop, types, spike_frames, counts, dt_ms=1.0, delay_ms=2.0):
    """Select active representatives; preserve their exact recorded firing times.

    Positions are supplied by the browser's schematic layout, not inferred anatomy.
    Edges are a sparse subset of actual directed, signed connections. An edge shows
    signal delivery after its source fires, not proof it caused the target to fire.
    """
    counts = np.asarray(counts).reshape(-1)
    group = np.minimum(np.asarray(hop), 2)
    selected, stages = [], []
    for stage, (label, limit) in enumerate(zip(STAGE_LABELS, STAGE_LIMITS)):
        candidates = np.flatnonzero((group == stage) & (counts > 0))
        order = np.lexsort((candidates, -counts[candidates]))
        chosen = candidates[order[:limit]]
        selected.extend(chosen.tolist())
        stages.append(dict(label=label, total_neurons=int((group == stage).sum()),
                           active_neurons=len(candidates), shown_neurons=len(chosen),
                           spikes=[0] * len(spike_frames)))

    selected = np.asarray(selected, dtype=np.int32)
    local = np.full(W.shape[0], -1, dtype=np.int32)
    local[selected] = np.arange(len(selected))
    nodes = [dict(neuron=int(neuron), type=str(types[neuron]),
                  stage=int(group[neuron]), count=int(counts[neuron]), spikes=[])
             for neuron in selected]
    seen = np.zeros(W.shape[0], dtype=bool)
    cumulative_spikes, cumulative_active, total = [], [], 0
    for step, fired in enumerate(spike_frames):
        fired = np.asarray(fired, dtype=np.int32)
        stage_counts = np.bincount(group[fired], minlength=len(stages))
        for stage, number in enumerate(stage_counts):
            stages[stage]["spikes"][step] = int(number)
        for node in local[fired]:
            if node >= 0:
                nodes[int(node)]["spikes"].append(round(step * dt_ms, 6))
        total += len(fired)
        seen[fired] = True
        cumulative_spikes.append(total)
        cumulative_active.append(int(seen.sum()))

    # W stores postsynaptic rows and presynaptic columns. Transpose so that
    # selecting the strongest outgoing edges does not reverse their direction.
    outgoing = W[selected][:, selected].T.tocsr()
    edges = []
    for source in range(len(selected)):
        start, end = outgoing.indptr[source:source + 2]
        targets, weights = outgoing.indices[start:end], outgoing.data[start:end]
        valid = (targets != source) & (weights != 0)
        targets, weights = targets[valid], weights[valid]
        order = np.lexsort((targets, -np.abs(weights)))[:3]
        for i in order:
            edges.append(dict(source=source, target=int(targets[i]),
                              excitatory=bool(weights[i] > 0)))

    return dict(duration_ms=len(spike_frames) * dt_ms, step_ms=dt_ms,
                delay_ms=delay_ms, nodes=nodes, edges=edges, stages=stages,
                cumulative_spikes=cumulative_spikes, cumulative_active=cumulative_active,
                shown_neurons=len(nodes), active_neurons=int((counts > 0).sum()),
                total_neurons=W.shape[0], layout="schematic")
