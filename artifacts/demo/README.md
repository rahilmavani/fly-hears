# Pretrained demo bundle

This is the `v2` reader and the exact graph it was trained against. The thirteen data files
total about 6 MB and are verified by SHA-256 in [manifest.json](manifest.json).

| File | Contents | License |
|---|---|---|
| `auditory.npz` | Derived 30,000-neuron MaleCNS v1.0 graph | CC BY 4.0 |
| `model.npz` | Trained reader, feature transform, and compatibility metadata | CC BY-SA 4.0 |
| `results.json` | Speaker-separated benchmark, confusion matrices, and model hash | CC BY-SA 4.0 |
| `recordings/*.wav` | Ten unmodified FSDD recordings | CC BY-SA 4.0 |
| `manifest.json` | File hashes and pinned dataset revision | CC BY-SA 4.0 |

See [third-party notices](../../THIRD_PARTY_NOTICES.md) for attribution and license links,
and [the model card](../../MODEL_CARD.md) for evaluation limits. The examples were used to train
the final reader; they are included for interaction and compatibility checks.

From the repository root:

```bash
uv sync --locked
uv run python -m scripts.setup_demo
uv run python -m app.server --run artifacts/demo --port 8001
```

Setup verifies all bundle files and compatibility, then copies only the graph into `data/`.
When the full FSDD is absent, the server serves these ten recordings. No original connectome
tables, training feature matrices, or user microphone recordings are included.

Treat this bundle as immutable. Train new experiments under `data/runs/`; replace the bundle
only when publishing a newly validated model, graph, report, and manifest together.
