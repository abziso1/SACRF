# SACRF

This repository contains the source code for SACRF, a multi-modal knowledge graph completion model.

## Dependencies
- Python 3.6+
- PyTorch 1.0+
- NumPy 1.17.2+
- tqdm 4.41.1+

## Data

The supported datasets are:

```text
MKG-Y
MKG-W
DB15K
```

Raw data should be placed under `src_data/`:

```text
src_data/
  MKG-Y/
    train
    valid
    test
    entity2id.txt
    relation2id.txt
  MKG-W/
    train
    valid
    test
    entity2id.txt
    relation2id.txt
  DB15K/
    train
    valid
    test
    entity2id.txt
    relation2id.txt
```

Multi-modal entity features should be placed under `embeddings/`:

```text
embeddings/
  MKG-Y-visual.pth
  MKG-Y-textual.pth
  MKG-W-visual.pth
  MKG-W-textual.pth
  DB15K-visual.pth
  DB15K-textual.pth
```

The multi-modal embeddings used in this project are from the M-Hyper repository: https://github.com/zjukg/M-Hyper.

## Preprocessing

Before training, preprocess the raw triples:

```bash
python process_datasets.py
```

This generates processed files under `data/`, including `train.pickle`, `valid.pickle`, `test.pickle`, and `to_skip.pickle`.

You only need to rerun preprocessing if the raw data in `src_data/` changes or if `data/` is removed.

## Training

Example commands:

```bash
python run.py --dataset MKG-Y --srank 128 --vrank 128 --trank 128 --gate_dim 128 --alpha_max 0.2 --optimizer AdamW --learning_rate 1e-3 --batch_size 500 --regularizer wN3 --reg 7e-2 --max_epochs 300 --valid 5 --seed 2026 --save
```

```bash
python run.py --dataset MKG-W --srank 128 --vrank 128 --trank 128 --gate_dim 128 --alpha_max 0.2 --optimizer AdamW --learning_rate 1e-3 --batch_size 500 --regularizer wN3 --reg 7e-2 --max_epochs 300 --valid 5 --seed 2026 --save
```

```bash
python run.py --dataset DB15K --srank 128 --vrank 128 --trank 128 --gate_dim 128 --alpha_max 0.2 --optimizer AdamW --learning_rate 1e-3 --batch_size 500 --regularizer wN3 --reg 7e-2 --max_epochs 300 --valid 5 --seed 2026 --save
```
