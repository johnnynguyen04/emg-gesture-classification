# EMG gesture classification

Classifying 53 hand gestures from surface EMG (NinaPro DB1) with a Random Forest baseline and a 1D CNN in PyTorch.

**Live demo:** https://emg-gestures.streamlit.app/

## Results

Trained on all 27 subjects, 1.2M windows, repetition-based split (reps 1–7 train, 8 val, 9–10 test).

| Model | Test accuracy | Macro F1 |
|---|---|---|
| Random Forest (Hudgins TD features) | **0.531** | 0.527 |
| 1D CNN (~60k params) | 0.352 | 0.348 |

The RF beat the CNN by 18 points. The Streamlit demo walks through why.

## Approach

- 200 ms windows, 100 ms overlap, labeled by stimulus at window center
- RF uses 6 time-domain features per channel (MAV, ZC, SSC, WL, RMS, variance)
- CNN: three Conv1D blocks, batchnorm + ReLU, adaptive pool, linear head
- Same split for both models, no augmentation, so the comparison is fair
- Split by repetition not by window so near-duplicate samples don't inflate test accuracy

## Stack

Python 3.11, PyTorch, scikit-learn, scipy, numpy, matplotlib, Streamlit. Dependencies pinned in `uv.lock`.

## Running locally

```bash
uv sync --extra dev
uv run pytest
uv run streamlit run streamlit_app/app.py
```

NinaPro DB1 `.mat` files go in `data/raw/`. The loader falls back to a synthetic stand-in if they aren't present, so the pipeline still runs end to end.

## What's next

Bigger CNN with augmentation, leave-one-subject-out cross-validation, and a try on DB2 or DB5 (raw EMG at 2 kHz where the bandpass filter actually applies).

## Citation

> Atzori M, Gijsberts A, Castellini C, Caputo B, Hager AGM, Elsig S, Giatsidis G, Bassetto F, Müller H. *Electromyography data for non-invasive naturally-controlled robotic hand prostheses*. Scientific Data 1, 140053 (2014).
