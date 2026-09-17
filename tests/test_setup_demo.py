import json
import tempfile
import unittest
from pathlib import Path

from scripts import artifacts
from scripts.setup_demo import install_graph, verify_bundle


class DemoSetupTests(unittest.TestCase):
    def test_install_is_repeatable_and_does_not_replace_another_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'bundled.npz'
            source.write_bytes(b'bundled graph')
            destination = root / 'data' / 'auditory.npz'
            install_graph(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            installed = destination.stat().st_mtime_ns
            install_graph(source, destination)
            self.assertEqual(destination.stat().st_mtime_ns, installed)
            destination.write_bytes(b'an existing experiment')
            with self.assertRaisesRegex(ValueError, 'different graph'):
                install_graph(source, destination)
            self.assertEqual(destination.read_bytes(), b'an existing experiment')

    def test_corrupted_bundle_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in ['auditory.npz', 'model.npz', 'results.json']:
                path = root / name
                path.write_bytes(b'original')
                files[name] = artifacts.file_hash(path)
            (root / 'manifest.json').write_text(json.dumps({'files': files}))
            (root / 'auditory.npz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'has changed'):
                verify_bundle(root)


if __name__ == '__main__':
    unittest.main()
