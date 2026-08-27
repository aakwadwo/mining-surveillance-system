# Backend

Full documentation lives at the project root:

- [`../README.md`](../README.md) — setup, running, API reference, troubleshooting
- [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — design decisions
- [`models/PROVENANCE.md`](models/PROVENANCE.md) — where the weights came from

## Quick start

```bash
cd ..
./.venv/bin/python backend/app.py     # http://127.0.0.1:5004
```

## Files

| File | Responsibility |
|---|---|
| `app.py` | HTTP routes only |
| `config.py` | paths, ports, thresholds, modes — the single source |
| `taxonomy.py` | the six detection classes — the single source |
| `detector.py` | model loading, inference, frame drawing |
| `models/` | trained weights + provenance |
| `data/`, `logs/` | runtime output (git-ignored contents) |
| `venv/` | the INHERITED, BROKEN environment. Kept as evidence; do not use. See ../README.md → Common errors. |

The rule this split enforces: **if a value appears in two files, one of
them is wrong.** The original single-file version defined the port in
three places and the class list in three places, all disagreeing.
