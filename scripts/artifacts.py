"""Record which code and graph produced each feature file and trained model."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
GRAPH_PATH = DATA / 'auditory.npz'
FORMAT_VERSION = 2


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(graph_path=None):
    graph_path = GRAPH_PATH if graph_path is None else graph_path
    source_files = ['encode.py', 'fly_lif.py', 'features.py', 'readout.py', 'simulate.py']
    return {
        'format': FORMAT_VERSION,
        'graph': file_hash(graph_path),
        'code': {name: file_hash(ROOT / 'scripts' / name) for name in source_files},
        'packages': {
            name: importlib.metadata.version(name)
            for name in ['numpy', 'scipy', 'librosa', 'scikit-learn']
        },
    }


def require_signature(metadata, expected):
    if metadata.get('signature') != expected:
        raise ValueError(
            'These files do not match the current encoder, simulator, or graph. '
            'Generate features and train again in a new run directory.'
        )


def read_metadata(data):
    if 'metadata' not in data:
        raise ValueError('Legacy artifact: no pipeline metadata. Generate fresh features and a new model.')
    return json.loads(str(data['metadata']))


def save_npz(path, metadata, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    try:
        with temporary.open('wb') as file:
            np.savez_compressed(file, metadata=json.dumps(metadata, sort_keys=True), **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
