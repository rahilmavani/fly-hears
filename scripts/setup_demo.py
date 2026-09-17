"""Verify the bundled demo and install its graph without replacing existing data."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from scripts import artifacts

BUNDLE = artifacts.ROOT / 'artifacts' / 'demo'


def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    required = {'auditory.npz', 'model.npz', 'results.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('The demo manifest is missing a required artifact.')
    for name, expected in manifest['files'].items():
        path = bundle / name
        if not path.resolve().is_relative_to(bundle.resolve()):
            raise ValueError(f'Invalid bundle path: {name}')
        if not path.is_file() or artifacts.file_hash(path) != expected:
            raise ValueError(f'Demo file is missing or has changed: {name}. Restore it from the repository.')
    with np.load(bundle / 'model.npz', allow_pickle=False) as data:
        artifacts.require_signature(artifacts.read_metadata(data),
                                    artifacts.signature(bundle / 'auditory.npz'))
    return manifest


def install_graph(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if artifacts.file_hash(destination) != artifacts.file_hash(source):
            raise ValueError(
                f'{destination} is a different graph. Use a separate checkout for the bundled demo '
                'or move your existing graph to a backup first.'
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        manifest = verify_bundle(BUNDLE)
        install_graph(BUNDLE / 'auditory.npz', artifacts.GRAPH_PATH)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Verified {len(manifest['files'])} bundled files. Demo graph is ready.")
    print('Start the demo: uv run python -m app.server --run artifacts/demo --port 8001')


if __name__ == '__main__':
    main()
