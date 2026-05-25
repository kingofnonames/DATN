# MCRGCN Improvements

## What was wrong in the original code

### 1. Hardcoded dataset (can't switch without editing source)
```python
# Original — GBM is hardcoded everywhere
data = sio.loadmat('GBM.mat')
features = data['GBM_Gene_Expression'].T
```
**Fix:** `Config` dataclass + `argparse` CLI. Switch with `--dataset BRCA`.

---

### 2. O(n²) nested loop for positive-pair matrix
```python
# Original — 248×248 = ~61,000 iterations just for GBM
for i in range(cora1.y[train_mask].shape[0]):
    for j in range(cora1.y[train_mask].shape[0]):
        if cora1.y[train_mask][i] == cora1.y[train_mask][j]:
            pos[i][j] = 1
```
**Fix:** Vectorised broadcast comparison — O(1) Python lines, GPU-accelerated:
```python
lbl = labels.view(n, 1)
pos = (lbl == lbl.t()).float()
```

---

### 3. Triplicated copy-paste for index mapping
The original repeated the same 10-line `index_*_dict` + edge-loading block three times verbatim.  
**Fix:** `load_edges(filename)` helper called three times.

---

### 4. No CLI / all hyperparameters buried in source
lr, epochs, seed, dataset name — all required editing the `.py` file.  
**Fix:** Full `argparse` interface. Examples:
```bash
python train_improved.py --dataset BRCA --epochs 200 --lr 5e-4 --patience 20
python train_improved.py --dataset GBM  --verbose
```

---

### 5. Shadowed built-in `dict`
```python
dict = dict()   # kills the built-in for the rest of the script
```
**Fix:** Removed; the variable was never actually used.

---

### 6. Logging via `print` + manual file open/close
Results were appended to files with raw `open(name, "a")` scattered everywhere, and stdout used `print`.  
**Fix:** Python `logging` module for console; a `_log_result()` / `_save_summary()` pair for file I/O.

---

### 7. No early stopping
Training always ran the full 120 epochs even if the loss plateaued.  
**Fix:** `patience` parameter (default 0 = disabled for backward compatibility).

---

### 8. Metric computation scattered inline
~30 lines of metric calls were copy-pasted inside every fold loop.  
**Fix:** `compute_metrics()` returns a clean `Dict[str, float]`.

---

### 9. Dead / commented-out code
~30 lines of commented survival-label saving code was left in the main loop.  
**Fix:** Removed.

---

### 10. No type hints or docstrings
**Fix:** Type hints on all public functions; module docstring explaining every improvement.

---

## Usage

```bash
# Install deps (same as original)
pip install torch torch-geometric scikit-learn scipy numpy

# Run on GBM (default)
python train_improved.py

# Run on BRCA with early stopping
python train_improved.py --dataset BRCA --patience 15

# Quick smoke test (1 repeat, 3 folds, 30 epochs)
python train_improved.py --dataset GBM --n_repeats 1 --n_splits 3 --epochs 30 --verbose
```

## Files unchanged from original repo
- `model/heco.py` — HeCo model architecture
- `model/contrast.py` — Contrastive loss
- `produce_adjacent_matrix.py` — adjacency matrix generation
- `produce _data.py` — edge CSV generation
- `BRCA.mat`, `GBM.mat` — datasets

## Potential future improvements (from paper's own future work section)
- Add attention mechanisms for omics fusion
- Incorporate additional omics (e.g. KEGG pathway features)
- Graph-level augmentation strategies beyond inter-omics comparison
- Hyperparameter search (optuna/ray-tune) given the sensitivity to threshold θ
