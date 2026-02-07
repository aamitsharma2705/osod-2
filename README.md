# OSOD-impl (Open-Set Object Detection scaffold)

End-to-end scaffold to reproduce the WACV’24 paper “Open-Set Object Detection by Aligning Known Class Representations.” It includes:
- Faster R-CNN with ConvNet-small backbone
- Object Focus RPN loss (centerness + objectness, Eq. 6–7)
- Stubs for semantic clustering (Eq. 2) and class decorrelation (Eq. 5)
- Hooks for entropy-thresholded evaluation (threshold = 0.85) and HMP
- Frozen CLIP text encoder (text branch only, 1024-D) for class prototypes

## 0) Prerequisites
- Python 3.11 (project targets 3.11)
- Poetry installed (`brew install poetry` or `pipx install poetry`)
- GPU recommended; CPU will be extremely slow

## 1) Setup
```bash
cd /Users/shivaninagpal/Documents/Amit/Project/vision/OSOD-impl
poetry install --with dev
```

## 2) Data
- Place VOC and COCO data under `data/` (mirroring the usual VOCdevkit/COCO layouts).
- Provide VOC–COCO T1/T2 split files under `data/splits/` (see `src/data/splits.py` for expected names).
- Add class-name text prompts for the CLIP encoder under `data/prompts/<split>.txt`.
- Prompt files added (VOC classes): `data/prompts/voc_coco_t1.txt`, `voc_coco_t2.txt`, `voc_sanity.txt`.
- Stub splits added for small/local runs (VOC2007 subset):
  - `data/splits/voc_coco_t1_train.txt` (50 ids), `voc_coco_t1_val.txt` (10 ids)
  - `data/splits/voc_coco_t2_train.txt` (50 ids), `voc_coco_t2_val.txt` (10 ids)
  These keep training light on this machine; replace with full T1/T2 lists for paper-scale runs.

Directory sketch:
```
data/
  VOC/
    VOC2007/...
    VOC2012/...
  COCO/
    train2017/...
    annotations/instances_train2017.json
  splits/
    voc_coco_t1_train.txt
    voc_coco_t1_val.txt
    voc_coco_t2_train.txt
    voc_coco_t2_val.txt
  prompts/
    voc_coco_t1.txt   # one class name per line
    voc_coco_t2.txt
    voc_sanity.txt    # VOC-only sanity prompts
```

## 3) Quick sanity run (VOC only, tiny subset)
```bash
poetry run python -m src.train \
  --data-root data/VOC \
  --split voc_sanity \
  --max-samples 4 \
  --num-iters 2
```
This exercises the pipeline, RPN with Object Focus loss, and CLIP text encoder wiring. It does not reproduce paper numbers.

## 3.1) Small T1/T2 stub run (light on CPU)
Uses the 50/10 stub splits we added; safe to run on this machine.
```bash
poetry run python -m src.train \
  --data-root data \
  --split voc_coco_t1 \
  --epochs 1 \
  --batch-size 2 \
  --lr 1e-4 \
  --entropy-thresh 0.85
```
Swap `voc_coco_t1` with `voc_coco_t2` to use the other stub.
Recommended now: run the command above to sanity-check the end-to-end pipeline on your current data and hardware.

## 4) Train on paper splits (skeleton)
```bash
poetry run python -m src.train \
  --data-root data \
  --split voc_coco_t1 \
  --epochs 12 \
  --batch-size 2 \
  --lr 1e-4 \
  --entropy-thresh 0.85
```
Notes:
- Semantic alignment (Eq. 2) and class decorrelation (Eq. 5) are stubbed; fill in losses in `src/models/losses/semantic_alignment.py` and `src/models/losses/class_decorrelation.py`.
- HMP metric and full evaluation loop are stubbed in `src/eval.py`.

## 5) Code map
- `src/config.py` — default hyperparameters.
- `src/clip_text.py` — frozen CLIP text encoder for class prototypes.
- `src/data/` — VOC loader and split utilities (extend for COCO and T1/T2).
- `src/models/backbones/convnet_small.py` — ConvNet-small backbone.
- `src/models/rpn/object_focus.py` — Object Focus RPN wrapper + centerness targets. Updated to:
  - Handle torchvision builds without `_cached_gt_indices` (falls back to background).
  - Align centerness logits with anchor count to avoid shape mismatches in loss.
- `src/models/roi_heads/semantic_cluster_head.py` — ROI semantic projection head (stubbed losses).
- `src/models/losses/` — semantic alignment, decorrelation, object focus, entropy helper.
- `src/models/build.py` — model factory that swaps in Object Focus RPN + semantic head.
- `src/train.py` — basic train loop wiring losses.
- `src/eval.py` — evaluation scaffold with entropy thresholding and HMP placeholder.

## 6) Next implementation steps
- Fill in Eq. (2) semantic alignment loss to align ROI features with CLIP text embeddings.
- Implement Eq. (5) class decorrelation regularizer over semantic cluster features.
- Complete HMP and entropy-thresholded evaluation in `src/eval.py`.
- Add COCO loader and proper VOC–COCO T1/T2 split readers in `src/data/splits.py`.
- See `nextsteps.md` for the full checklist and order of work.

## Recent run readiness
- Stub splits and prompts are provided for light local runs (voc_coco_t1/voc_coco_t2).
- Training command tested (short run): \
  `poetry run python -m src.train --data-root data --split voc_coco_t1 --epochs 1 --batch-size 2 --lr 1e-4 --entropy-thresh 0.85 --max-samples 50 --num-iters 50`
- Optional eval/checkpointing flags:
  - `--eval-every 1` runs entropy/HMP eval each epoch (placeholder metrics until unknown labels are wired).
  - `--checkpoint-dir checkpoints` and `--resume checkpoints/latest.pth` to save/resume.
- Dataset builder supports VOC+COCO composite for T1/T2 (requires COCO data present under `data/COCO`).
- Losses: Semantic alignment now uses normalized ROI projections vs CLIP text embeds (cosine/temperature logits, CE); decorrelation regularizes projected ROI features (off-diagonal correlation penalty).
- Semantic alignment/ decorrelation now wired: ROI features exposed, background filtered, and losses applied; CLIP dim set to 512 to match ViT-B/32 text encoder.
- RPN/object focus fixes: centerness aligned to anchors; safe handling of missing `_cached_gt_indices`.
- Torchvision compatibility fixes: RPN feature dict wrapping; ROI loss fallback handles different torchvision APIs.
- Data splits: global train/trainval/val files generated for VOC2007/2012 so loaders work.

Status vs paper:
- Object Focus RPN (Eq. 6–7): implemented and active.
- Semantic alignment/decorrelation: implemented as cosine/temperature CE + off-diagonal decorrelation; weights/hparams still need to be set to paper values.
- Evaluation (entropy threshold/HMP): still stubbed; not paper-complete.
- Data splits: stub T1/T2 lists; full paper splits and COCO loader still pending.

## Recent changes and impact
- Added prompt files under `data/prompts/` (`voc_coco_t1.txt`, `voc_coco_t2.txt`, `voc_sanity.txt`) with VOC class names to enable CLIP text embeddings for sanity and T1/T2 runs.
- Hardened `src/models/rpn/object_focus.py` to:
  - Work with torchvision builds lacking `_cached_gt_indices` (prevents AttributeError during RPN forward).
  - Match centerness logits to the anchor count (fixes shape mismatch in L1 loss).
Impact: Sanity training runs now proceed without the earlier RPN centerness/anchor shape errors; CLIP prompt loading works out of the box for VOC/T1/T2 splits.
