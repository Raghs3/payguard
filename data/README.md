# Data

Raw and processed data are **not committed to git** (see `.gitignore`) — too large, and mostly public/redistributable data anyway. This file is the source of truth for getting the data locally.

## Dataset

- **IEEE-CIS Fraud Detection** — [Kaggle link](https://www.kaggle.com/c/ieee-fraud-detection/data)
- Download the competition files and place them under `data/raw/`.

## Setup

```bash
# from repo root, with Kaggle API configured
kaggle competitions download -c ieee-fraud-detection -p data/raw/
cd data/raw && unzip ieee-fraud-detection.zip
```

## Schema notes

[Fill in as the team explores the data — key columns, target variable, known quirks/imbalance ratio, etc.]

## Processed data

Feature-engineered datasets produced by `src/features/` land in `data/processed/` — also gitignored. Document the transformation steps in `src/features/README.md` if that folder grows large, so processed data is reproducible from raw + code alone.
