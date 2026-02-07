import torch
import torch.nn as nn
from torchvision.models.detection.rpn import RegionProposalNetwork


class CenternessHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, features):
        x = self.conv(features)
        return x.flatten(start_dim=1)


def compute_centerness_targets(
    proposals: torch.Tensor, gt_boxes: torch.Tensor, matched_gt_indices: torch.Tensor
) -> torch.Tensor:
    """
    FCOS-style centerness targets, used in Eq. (6)–(7).
    """
    device = proposals.device
    N = proposals.size(0)
    centerness = torch.zeros(N, device=device)

    fg_mask = matched_gt_indices >= 0
    if fg_mask.sum() == 0:
        return centerness

    fg_proposals = proposals[fg_mask]
    fg_gt = gt_boxes[matched_gt_indices[fg_mask]]

    px = (fg_proposals[:, 0] + fg_proposals[:, 2]) / 2.0
    py = (fg_proposals[:, 1] + fg_proposals[:, 3]) / 2.0

    l = px - fg_gt[:, 0]
    r = fg_gt[:, 2] - px
    t = py - fg_gt[:, 1]
    b = fg_gt[:, 3] - py

    eps = 1e-6
    l = torch.clamp(l, min=eps)
    r = torch.clamp(r, min=eps)
    t = torch.clamp(t, min=eps)
    b = torch.clamp(b, min=eps)

    centerness_fg = torch.sqrt(
        (torch.min(l, r) / torch.max(l, r)) * (torch.min(t, b) / torch.max(t, b))
    )
    centerness[fg_mask] = centerness_fg
    return centerness


class ObjectFocusLoss(nn.Module):
    """
    Eq. (6): L1 centerness loss
    Eq. (7): Object Focus = sqrt(L_C * L_Obj)
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(
        self,
        objectness_loss: torch.Tensor,
        centerness_logits: torch.Tensor,
        centerness_targets: torch.Tensor,
    ) -> torch.Tensor:
        L_C = self.l1(centerness_logits, centerness_targets)
        return torch.sqrt(L_C * objectness_loss)


class RPNWithObjectFocus(nn.Module):
    """
    Wrapper that replaces RPN objectness loss with Object Focus Loss.
    """

    def __init__(self, base_rpn: RegionProposalNetwork):
        super().__init__()
        self.rpn = base_rpn
        self.centerness_head = CenternessHead(self.rpn.head.conv[0].out_channels)
        self.object_focus_loss = ObjectFocusLoss()

    def forward(self, images, features, targets=None):
        boxes, losses = self.rpn(images, features, targets)

        if self.training:
            objectness_loss = losses.pop("loss_objectness")
            box_loss = losses.pop("loss_rpn_box_reg")

            feature_map = list(features.values())[0]
            centerness_logits = self.centerness_head(feature_map)  # (B, H*W)

            # Match anchor granularity: repeat per anchor at each location.
            anchors_per_loc = self.rpn.anchor_generator.num_anchors_per_location()[0]
            centerness_logits = (
                centerness_logits.unsqueeze(-1)
                .repeat(1, 1, anchors_per_loc)  # (B, H*W, A)
                .reshape(-1)
            )

            anchors = self.rpn.anchor_generator(images, list(features.values()))
            flat_anchors = torch.cat(anchors, dim=0)

            # Some torchvision builds do not expose _cached_gt_indices; fall back to all background.
            matched_gt_indices = getattr(
                self.rpn, "_cached_gt_indices", None
            )
            if matched_gt_indices is None:
                matched_gt_indices = torch.full(
                    (flat_anchors.shape[0],), -1, device=flat_anchors.device
                )

            if len(targets) > 0 and any(t["boxes"].numel() > 0 for t in targets):
                gt_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
            else:
                gt_boxes = torch.zeros((0, 4), device=flat_anchors.device)

            centerness_targets = compute_centerness_targets(
                flat_anchors, gt_boxes, matched_gt_indices
            )

            obj_focus_loss = self.object_focus_loss(
                objectness_loss, centerness_logits, centerness_targets
            )

            losses = {
                "loss_rpn_object_focus": obj_focus_loss,
                "loss_rpn_box_reg": box_loss,
            }

        return boxes, losses

