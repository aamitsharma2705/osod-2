"""
Scene 1: Closed-Set Object Detection Visualization
Paper: Open-Set Object Detection by Aligning Known Class Representations (WACV 2024)
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms

from src.data.voc import VOCDataset, VOC_CLASSES

# ----------------------------
# Load Faster R-CNN (Baseline)
# ----------------------------
print("[INFO] Loading Faster R-CNN model...")
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# ----------------------------
# Load VOC 2007 Test Dataset
# ----------------------------
print("[INFO] Loading VOC 2007 test dataset...")
DATA_ROOT = "/Users/shivaninagpal/Documents/Amit/Project/vision/OSOD-impl/data/VOC"

dataset = VOCDataset(
    DATA_ROOT,
    year="2007",
    image_set="test"
)

to_tensor = transforms.ToTensor()

# ----------------------------
# Pick One Image
# ----------------------------
image, target = dataset[25]

# Ensure tensor
if not torch.is_tensor(image):
    image_t = to_tensor(image)
else:
    image_t = image

# ----------------------------
# Run Inference
# ----------------------------
print("[INFO] Running inference...")
with torch.no_grad():
    outputs = model([image_t])[0]

# ----------------------------
# Visualization
# ----------------------------
print("[INFO] Visualizing detections...")

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

    # Safety: skip invalid labels
    if label_id >= len(VOC_CLASSES):
        continue

    x1, y1, x2, y2 = box.tolist()

    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor="lime",
        facecolor="none"
    )
    ax.add_patch(rect)

    ax.text(
        x1,
        y1 - 5,
        f"{VOC_CLASSES[label_id]} ({score:.2f})",
        fontsize=10,
        color="black",
        backgroundcolor="lime"
    )

plt.title("Scene 1: Closed-Set Detection (VOC Only)")
plt.show()

print("[DONE] Scene 1 complete.")
