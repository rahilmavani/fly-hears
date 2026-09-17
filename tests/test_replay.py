import unittest

import numpy as np
import scipy.sparse as sp

from app.replay import build_replay


class ReplayTests(unittest.TestCase):
    def test_recorded_times_counts_and_connection_direction(self):
        # Directed chain plus a feedback edge. Rows are targets, columns sources.
        W = sp.csr_matrix(([2., -3., 4.], ([1, 2, 1], [0, 1, 2])), shape=(4, 4))
        frames = [np.array(f, dtype=int) for f in [[0], [1], [2], [0, 2], []]]
        replay = build_replay(W, np.array([0, 1, 2, 2]), np.array(['ear', 'relay', 'deep', 'quiet']),
                              frames, np.array([2, 1, 2, 0]), dt_ms=0.5)
        self.assertEqual(replay['duration_ms'], 2.5)
        self.assertEqual(replay['cumulative_spikes'], [1, 2, 3, 5, 5])
        self.assertEqual(replay['cumulative_active'], [1, 2, 3, 3, 3])
        nodes = replay['nodes']
        self.assertEqual({n['neuron']: n['spikes'] for n in nodes},
                         {0: [0., 1.5], 1: [0.5], 2: [1., 1.5]})
        edges = {(nodes[e['source']]['neuron'], nodes[e['target']]['neuron'], e['excitatory'])
                 for e in replay['edges']}
        self.assertEqual(edges, {(0, 1, True), (1, 2, False), (2, 1, True)})
        self.assertEqual(replay['stages'][0]['spikes'], [1, 0, 0, 1, 0])
        self.assertEqual(replay['stages'][1]['spikes'], [0, 1, 0, 0, 0])
        self.assertEqual(replay['stages'][2]['spikes'], [0, 0, 1, 1, 0])

    def test_sampling_keeps_full_network_totals(self):
        n = 250
        hop = np.repeat([0, 1, 2], [60, 90, 100])
        replay = build_replay(sp.csr_matrix((n, n)), hop, np.full(n, 'cell'),
                              [np.arange(n)], np.ones(n, dtype=int))
        self.assertEqual(replay['shown_neurons'], 144)
        self.assertEqual(replay['active_neurons'], n)
        self.assertEqual(replay['cumulative_spikes'], [n])
        self.assertEqual([s['shown_neurons'] for s in replay['stages']], [32, 48, 64])
        self.assertEqual([s['spikes'] for s in replay['stages']], [[60], [90], [100]])

    def test_silent_recording_has_no_fabricated_nodes_or_edges(self):
        replay = build_replay(sp.eye(3, format='csr'), np.array([0, 1, 2]), np.array(['a', 'b', 'c']),
                              [np.array([], dtype=int)] * 3, np.zeros(3, dtype=int))
        self.assertEqual(replay['nodes'], [])
        self.assertEqual(replay['edges'], [])
        self.assertEqual(replay['cumulative_spikes'], [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
