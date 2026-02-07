"""
Scene 3: Proposed OSOD Model Visualization
Using your own implementation from the 'osod-2' repo.

Purpose:
- Use your trained OSOD model (semantic clustering + class decorrelation
  + object focus) to visualize open-set predictions.
- Show how your model handles unknown objects compared to Scene 2.

This script assumes:
- You have a trained checkpoint in ./checkpoints/
- Your VOC-COCO open-set split and prompts are configured
- `src` module is importable via current path
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

# Make src importable
sys.path.append(os.getcwd())

from src.config import get_cfg
from src.data.splits import OpenSetVOCCOCOSplit
from src.data.prompts import load_prompts
from src.models.build import build_model
from src.clip_text import CLIPTextEncoder

# ----------------------------
# CONFIGURATION
# ----------------------------
print("[INFO] Loading configuration...")
cfg = get_cfg()
cfg.DATA.ROOT = "/Users/shivaninagpal/Documents/Amit/Project/vision/OSOD-impl/data"
cfg.MODEL.CHECKPOINT = "/Users/shivaninagpal/Documents/Amit/Project/vision/OSOD-impl/checkpoints/osod_model.pth"
cfg.DATA.SPLIT = "voc_coco_t1_val"   # or voc_coco_t2_val

# Load class prompts for CLIP
print("[INFO] Loading CLIP text prompts...")
class_prompts = load_prompts(os.path.join(cfg.DATA.ROOT, "prompts", cfg.DATA.SPLIT + ".txt"))

# ----------------------------
# DATASET LOAD
# ----------------------------
print("[INFO] Building VOC-COCO open-set dataset...")
dataset = OpenSetVOCCOCOSplit(
    root=cfg.DATA.ROOT,
    split_name=cfg.DATA.SPLIT,
    unknown_only=True   # only unknown objects in this run
)

# Pick a representative unknown image (same index as Scene 2)
img, meta = dataset[10]

# Image to tensor
img_tensor = torch.tensor(img).permute(2,0,1) / 255.0
img_batch = img_tensor.unsqueeze(0)

# ----------------------------
# MODEL BUILD + LOAD
# ----------------------------
print("[INFO] Building OSOD model...")
model, _, _ = build_model(cfg)

print(f"[INFO] Loading checkpoint from {cfg.MODEL.CHECKPOINT}...")
checkpoint = torch.load(cfg.MODEL.CHECKPOINT, map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()

# ----------------------------
# CLIP Text Encoder (only if needed for prompts)
# ----------------------------
print("[INFO] Initializing CLIP text encoder...")
clip_encoder = CLIPTextEncoder()
clip_encoder.eval()

# ----------------------------
# RUN INFERENCE
# ----------------------------
print("[INFO] Running proposed OSOD inference...")
with torch.no_grad():
    outputs = model(img_batch)[0]

# ----------------------------
# VISUALIZATION
# ----------------------------
print("[INFO] Visualizing proposed OSOD output...")

fig, ax = plt.subplots(1, figsize=(10,8))
ax.imshow(img_tensor.permute(1,2,0))
ax.axis("off")

for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
    if score < cfg.EVAL.SCORE_THR:
        continue

    x1, y1, x2, y2 = box.tolist()

    # Label name (your dataset likely maps known vs unknown)
    label_name = dataset.label_map[label.item()]

    color = "yellow" if dataset.is_known(label.item()) else "red"

    rect = patches.Rectangle(
        (x1,y1), x2-x1, y2-y1,
        linewidth=2, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(
        x1, y1 - 5,
        f"{label_name} ({score:.2f})",
        color="black", backgroundcolor=color
    )

plt.title("Scene 3: Proposed OSOD Model Output")
plt.show()

print("[DONE] Scene 3 complete.")
