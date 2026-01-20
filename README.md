# Roof Area

Utilities for roof area inference, evaluation, and training.

## Setup & Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional training dependencies:

```bash
pip install -e '.[train]'
```

## CLI-Beispiele

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

## Legal-Open-Data Hinweis

Die Nutzung von Geobasisdaten der Landesvermessung sowie Daten des Bundesamts fuer Kartographie und Geodaesie (BKG) kann Lizenz- und Nutzungspflichten enthalten. Bitte pruefen Sie die jeweils gueltigen Open-Data-Lizenzen der zustaendigen Stellen, bevor Sie Daten weiterverwenden oder veroeffentlichen.
