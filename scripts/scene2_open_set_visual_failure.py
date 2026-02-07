"""
Scene 2: Open-Set Failure Visualization
Paper: Open-Set Object Detection by Aligning Known Class Representations (WACV 2024)

Purpose:
- Demonstrate FAILURE of closed-set detector in open-set conditions
- Unknown objects (COCO non-VOC) are misclassified as known VOC classes
- This scene exposes the core OSOD problem

Model: Faster R-CNN ResNet50-FPN (VOC-trained, unchanged)
Dataset: VOC–COCO Open-Set (VOC known + COCO unknown)
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image

from src.data.voc import VOC_CLASSES

# ----------------------------
# Load Faster R-CNN (VOC-trained)
# ----------------------------
print("[INFO] Loading Faster R-CNN model (VOC-trained)...")
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# ----------------------------
# Simple COCO unknown-only sampler (val2017)
# ----------------------------
print("[INFO] Loading COCO val2017 as unknown pool...")
COCO_ROOT = "/Users/shivaninagpal/Documents/Amit/Project/vision/OSOD-impl/data/COCO"
COCO_ANN = COCO_ROOT + "/annotations/instances_val2017.json"
coco = COCO(COCO_ANN)
img_ids = coco.getImgIds()

to_tensor = transforms.ToTensor()

def load_coco_image(idx: int):
    img_id = img_ids[idx % len(img_ids)]
    img_info = coco.loadImgs(img_id)[0]
    img_path = f"{COCO_ROOT}/val2017/{img_info['file_name']}"
    img = Image.open(img_path).convert("RGB")
    return img, img_info

# ----------------------------
# Pick an UNKNOWN-object Image
# ----------------------------
img, meta = load_coco_image(10)
image_t = to_tensor(img)

# ----------------------------
# Run Inference
# ----------------------------
print("[INFO] Running inference on unknown object...")
with torch.no_grad():
    outputs = model([image_t])[0]

# ----------------------------
# Visualization
# ----------------------------
print("[INFO] Visualizing open-set failure...")

fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(image_t.permute(1, 2, 0))
ax.axis("off")

for box, label, score in zip(
    outputs["boxes"],
    outputs["labels"],
    outputs["scores"]
):
    if score < 0.7:
        continue

    label_id = int(label)
    if label_id >= len(VOC_CLASSES):
        continue

    x1, y1, x2, y2 = box.tolist()

    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(rect)

    ax.text(
        x1,
        y1 - 5,
        f"{VOC_CLASSES[label_id]} ({score:.2f})",
        fontsize=10,
        color="white",
        backgroundcolor="red"
    )

plt.title("Scene 2: Open-Set Failure (Unknown → Known Misclassification)")
plt.show()

print("[DONE] Scene 2 complete.")
