# EMG gesture classification

Classifying 53 hand gestures from surface EMG (NinaPro DB1). Random Forest baseline + 1D CNN in PyTorch.

I'm a Data Science major at UCF. I wanted to take a biosignal project end to end: load the raw signal, preprocess it properly, train two classifiers worth comparing, and write up where each one wins.

## What this is

NinaPro DB1 is a public sEMG dataset. 27 subjects, 10 channels recorded from the forearm, 52 hand/wrist movements plus rest. Every subject performs every gesture 10 times. The goal here is to take a 200 ms window of EMG and predict which gesture the subject is doing.

The pipeline trains and compares two models on the same split, so the comparison is honest:

1. **Random Forest** on the Hudgins time-domain features: MAV, ZC, SSC, WL, RMS, variance. 60 features per window. This is what most myoelectric control papers use as a baseline.
2. **1D CNN** trained end to end on the raw windowed signal. Three Conv1D blocks, batchnorm + ReLU, adaptive pooling, dropout, linear head.

## A note on DB1 specifically

DB1 is sampled at 100 Hz on Otto Bock electrodes that output a pre-rectified envelope, not raw EMG. That matters because a lot of EMG tutorials tell you to apply a 20–450 Hz bandpass. At 100 Hz that's impossible (Nyquist is 50 Hz), and on an already-rectified signal it's pointless anyway. I left the bandpass function in `src/preprocess.py` for portability to DB2 and DB5 (raw EMG at 2 kHz), but it's skipped for DB1, and calling it on a 100 Hz signal raises with a clear message about why.

## Project layout

```
src/ data loader, preprocessing, features, model, training, evaluation
notebooks/ 5 notebooks: exploration → preprocess → RF → CNN → analysis
streamlit_app/ upload a window, see the predicted gesture and top-5 probs
tests/ pytest smoke tests for the loader, windowing, and model forward pass
data/raw/ NinaPro DB1 .mat files go here (gitignored)
data/processed/ windowed numpy arrays after notebook 02 (gitignored)
results/ metrics JSON and confusion-matrix figures
```

## Approach

**Preprocessing.** Rectify (already done by the hardware on DB1, idempotent here), slide 200 ms windows with 100 ms overlap, label by the stimulus at the window center, drop any window that crosses a gesture boundary so no window has an ambiguous label.

**Splitting.** Within-subject by repetition. Reps 1–6 train, 7–8 val, 9–10 test. Splitting by repetition instead of by window index is what keeps train and test from seeing nearly identical samples. That's a real footgun in EMG papers — overlapping windows from the same repetition can be near-duplicates and inflate test accuracy by a lot.

**RF baseline.** sklearn's RandomForestClassifier with 200 trees and default everything else. 60 features per window.

**CNN.** Channels-first input `(batch, 10, 20)`. Three Conv1D layers (32 → 64 → 128 channels, kernel 5), batchnorm + ReLU after each, max-pool 2× after the first two, adaptive average pool at the end, dropout 0.3, linear → 53 logits. Adam at 1e-3, cross-entropy loss, early stopping on val accuracy.

**Cross-subject (v2).** Within-subject is the easier setting and the one most papers report first. Leave-one-subject-out is the next milestone; the code path is in `src/data.py` but I haven't run it yet.

## Results

Trained on all 27 subjects, 1.2M windows, repetition-based split (reps 1–7 train, 8 val, 9–10 test). RF used 200 trees, CNN was trained for 30 epochs at batch 512, lr 2e-3.

| Model | Test accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Random Forest (TD features) | **0.531** | 0.527 | 0.527 |
| 1D CNN | 0.352 | 0.348 | 0.347 |

The RF beat the CNN by 18 points. That was not what I expected going in.

### Why I think the CNN lost

A few things, in roughly the order they probably matter:

1. **DB1's signal is already a rectified envelope.** The Otto Bock electrodes do hardware rectification and smoothing before the signal hits the 100 Hz ADC. Most of the high-frequency information a CNN would normally learn to extract is gone before the sample is recorded. The Hudgins TD features (MAV, RMS, etc.) are essentially what's left, so the RF is operating on close to the same representation a CNN would have to reconstruct.
2. **20 timesteps is short.** 200 ms × 100 Hz = 20 samples per window. Three Conv1D + pool layers chew that down to a single timestep before the linear head. Not a lot of room for hierarchical features.
3. **60k parameters is small for 53 classes.** Average about 1.1k parameters per class. Published DB1 CNN papers that beat the TD baseline usually run >500k parameters with augmentation.
4. **No augmentation.** I deliberately skipped this to keep the comparison fair — RF doesn't get augmentation either — but a real CNN-vs-RF comparison would augment the CNN's training set with window jitter, channel dropout, and amplitude scaling, which all hurt the RF much less than they help the CNN.

The five classes the CNN struggled most on (recall < 0.17) are all class IDs 31, 32, 40, 41, 42 — every one of them is in the E3 exercise set, which is functional grasps (e.g. handle wrenching, lateral grasp, tripod pinch). Those are the most muscle-coordinated, multi-channel patterns in the dataset, and they're where the CNN's lack of capacity hurts most.

Confusion matrices and metric JSONs are in [results/](results/).

### What this tells me

The "CNN > classical features" story is real in EMG, but it depends on signal quality, model capacity, and augmentation. On a 100 Hz envelope dataset with a small model and no augmentation, the classical pipeline still wins. That's an honest result and a reasonable starting point for a v2 that addresses the four things above.

## What's next

- **Bigger CNN + augmentation.** Same dataset, ~500k params, window jitter and amplitude scaling. I want to see if the CNN can clear the RF baseline once given a fair shot at the problem.
- **Cross-subject (leave-one-subject-out).** Within-subject is easy mode. Real clinical use has to generalize to a person the model has never seen, and the literature shows accuracy drops 30+ points when you do that. Seeing that drop on my own pipeline, then trying per-subject normalization to claw some of it back, is the real test.
- **Try DB2 or DB5.** Both are raw EMG at 2 kHz instead of a pre-rectified envelope at 100 Hz. The bandpass function in `src/preprocess.py` becomes useful again, and the CNN should have a much fairer shot.

## Tech stack

Python 3.11, PyTorch, scikit-learn, scipy, numpy, pandas, matplotlib, seaborn, Streamlit. Dependencies are managed with [uv](https://github.com/astral-sh/uv) and pinned in `uv.lock`.

## Running it locally

```bash
# install
uv sync --extra dev

# get the data
# Register at http://ninapro.hevs.ch/instructions/DB1.html, download DB1,
# and drop S*_A1_E*.mat into data/raw/. The pipeline falls back to a
# synthetic stand-in if no real files are present, so you can smoke-test
# the full pipeline without the dataset.

# run the tests
uv run pytest

# run notebooks in order
uv run jupyter lab

# launch the demo (needs a trained model at results/metrics/cnn_best.pt)
uv run streamlit run streamlit_app/app.py
```

## Citation

If you use this code or build on it, please cite the NinaPro authors:

> Atzori M, Gijsberts A, Castellini C, Caputo B, Hager AGM, Elsig S, Giatsidis G, Bassetto F, Müller H. *Electromyography data for non-invasive naturally-controlled robotic hand prostheses*. Scientific Data 1, 140053 (2014).

Dataset use is subject to the NinaPro license terms on their site.

## Status

- Full pipeline ran end to end on all 27 DB1 subjects
- RF beat the small vanilla CNN by 18 points; writeup in Results
- Streamlit demo runs locally against the trained CNN checkpoint
- Cross-subject evaluation, bigger CNN, and augmentation are the v2 milestones
