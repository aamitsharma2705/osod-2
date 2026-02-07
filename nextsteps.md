# Next Steps to Complete Paper Implementation

## Losses & Heads
- Implement Eq. (2) semantic alignment loss in `src/models/losses/semantic_alignment.py` (cosine/temperature with labels) and wire ROI features/labels in `train.py`.
- Implement Eq. (5) class decorrelation in `src/models/losses/class_decorrelation.py` (orthogonality/feature decorrelation) and apply to projected ROI features.
- Integrate Eq. (2) and Eq. (5) into the training loop with correct weights from the paper; update `config.py` for these weights.

## Data & Splits
- Add proper VOC–COCO T1/T2 split readers in `src/data/splits.py` and replace stub lists with full paper splits.
- Add a COCO dataset loader and compose VOC + COCO datasets per T1/T2 protocol.
- Verify CLIP prompt files for all classes used in T1/T2; extend prompts if needed.

## Evaluation
- Complete entropy-thresholded evaluation in `src/eval.py` using threshold = 0.85 (paper). ✔ (GT matching for PK added; unknown still placeholder)
- Implement HMP (harmonic mean of known/unknown precision) in `src/eval.py` and hook into eval loop. ✔ (unknown still placeholder)
- Add logging of known/unknown precision and HMP during validation. ✔

## Training Loop & Checkpoints
- Add checkpoint saving (e.g., end of each epoch) and resume support in `src/train.py`.
- Add basic logging (loss components, LR) and optional tensorboard/CSV logging.
- Expose CLI args for loss weights, temperature, and checkpoint paths.

## RPN & ROI plumbing
- Verify RPN anchor/centerness target shapes across torchvision versions (current fixes are in place; keep).
- Expose ROI pooled features and labels cleanly for semantic/decorrelation losses instead of using placeholder keys.

## Config & Hyperparameters
- Encode paper hyperparameters (loss weights, temperature, optimizer, LR schedule, epochs) in `config.py`.
- Add CLI overrides for critical hyperparameters.

## Validation & Sanity
- Add a small smoke-test script that runs a few iterations on tiny splits to catch regressions.
- Add a validation routine to run after each epoch with entropy-thresholded metrics and HMP.

## Docs
- Update `README.md` once losses/eval are complete (how to run full T1/T2, where checkpoints are saved, expected metrics).

