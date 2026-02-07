import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from src.models.backbones.convnet_small import ConvNetSmall
from src.models.rpn.object_focus import RPNWithObjectFocus
from src.models.roi_heads.semantic_cluster_head import SemanticClusterHead
from torchvision.models.detection.roi_heads import fastrcnn_loss as tv_fastrcnn_loss


class FasterRCNNSemantic(nn.Module):
    """
    Faster R-CNN with:
    - ConvNet-small backbone
    - Object Focus RPN
    - Semantic clustering head atop ROI features
    """

    def __init__(self, num_classes: int, embed_dim: int = 512, temperature: float = 0.07):
        super().__init__()
        backbone = ConvNetSmall()
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )
        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )
        # Swap RPN
        self.detector.rpn = RPNWithObjectFocus(self.detector.rpn)
        # Semantic head
        self.semantic_head = SemanticClusterHead(
            in_dim=self.detector.roi_heads.box_head.fc7.out_features,
            embed_dim=embed_dim,
            temperature=temperature,
        )

    def forward(self, images, targets=None, class_embeds=None):
        # In eval, defer to the built-in detector forward.
        if not self.training:
            return self.detector(images, targets)

        # Training path: replicate ROI forward to expose ROI features/labels.
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.detector.transform(images, targets)
        features = self.detector.backbone(images.tensors)
        if not isinstance(features, dict):
            features = {"0": features}

        proposals, proposal_losses = self.detector.rpn(images, features, targets)

        proposals, matched_idxs, labels, regression_targets = (
            self.detector.roi_heads.select_training_samples(proposals, targets)
        )
        # labels is a list per image; stack for loss computation
        labels_tensor = torch.cat(labels, dim=0)

        box_features = self.detector.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes
        )
        box_features = self.detector.roi_heads.box_head(box_features)
        class_logits, box_regression = self.detector.roi_heads.box_predictor(
            box_features
        )

        # Torchvision compatibility: use helper if methods not present
        if hasattr(self.detector.roi_heads, "fastrcnn_loss"):
            loss_classifier, loss_box_reg = self.detector.roi_heads.fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
        elif hasattr(self.detector.roi_heads, "loss_classifier"):
            loss_classifier, loss_box_reg = self.detector.roi_heads.loss_classifier(
                class_logits, box_regression, labels, regression_targets
            )
        else:
            # tv helper expects lists for labels/regression_targets
            loss_classifier, loss_box_reg = tv_fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )

        losses = {
            **proposal_losses,
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg,
        }

        semantic_feats = {"roi_feats": box_features.detach(), "labels": labels_tensor.detach()}
        return losses, semantic_feats


def build_model(num_classes: int, embed_dim: int = 512, temperature: float = 0.07):
    return FasterRCNNSemantic(num_classes=num_classes, embed_dim=embed_dim, temperature=temperature)

