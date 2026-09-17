"""Check the data and experiment files without simulating or training."""

import argparse
import shutil
from pathlib import Path

import numpy as np

from scripts import artifacts, features
from scripts.simulate import recordings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Also check an existing experiment directory')
    args = parser.parse_args()
    if not artifacts.GRAPH_PATH.exists():
        parser.error('Missing data/auditory.npz. Download the connectome and run scripts.subgraph first.')
    graph = features.load_graph(artifacts.GRAPH_PATH)
    paths, labels, speakers = recordings(artifacts.DATA / 'fsdd' / 'recordings')
    if not np.array_equal(np.unique(labels), np.arange(10)):
        parser.error('The dataset must contain all ten digits.')
    for speaker in np.unique(speakers):
        if not np.array_equal(np.unique(labels[speakers == speaker]), np.arange(10)):
            parser.error(f'Speaker {speaker} does not have all ten digits.')
    print(f'Network: {graph.size:,} neurons and {graph.weights.nnz:,} connections')
    print(f'Dataset: {len(paths):,} recordings; speakers: {", ".join(np.unique(speakers))}')
    print(f'Disk available: {shutil.disk_usage(artifacts.DATA).free / 1024**3:.1f} GiB')
    print('Code, graph, and package signatures are available.')
    current_signature = artifacts.signature()
    if args.run:
        if not args.run.is_dir():
            print(f'New experiment directory: {args.run}')
        else:
            for path in sorted(args.run.glob('*.npz')):
                with np.load(path, allow_pickle=False) as data:
                    artifacts.require_signature(artifacts.read_metadata(data), current_signature)
                print(f'Compatible: {path.name}')
    print('Preflight passed. No simulation or training was run.')


if __name__ == '__main__':
    main()
