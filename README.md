# Roof Area

Utilities for roof area inference, evaluation, and training.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional training dependencies:

```bash
pip install -e '.[train]'
```

## CLI

Run inference:

```bash
roof-area infer --threshold 0.6 --tile-size 512
```

Run evaluation:

```bash
roof-area eval --min-area-m2 10
```

Run training (optional):

```bash
roof-area train --seed 123
```
