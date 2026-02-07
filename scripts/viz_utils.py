import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_detections(image, predictions, score_thr=0.5, title=""):
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)
    ax.axis("off")

    for box, label, score in zip(
        predictions["boxes"],
        predictions["labels"],
        predictions["scores"]
    ):
        if score < score_thr:
            continue

        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5,
            f"{label} ({score:.2f})",
            color="yellow",
            fontsize=10,
            backgroundcolor="black"
        )

    plt.title(title)
    plt.show()
