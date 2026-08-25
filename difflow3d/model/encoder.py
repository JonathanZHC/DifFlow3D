"""PointConv encoder and reusable encoded-frame representation."""

from typing import NamedTuple

import torch
import torch.nn as nn

from .pointconv import Conv1d, PointConv, PointConvD, index_points_group, knn_point_with_relative

def _self_knn_context(
    xyz_channel_first: torch.Tensor,
    neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return self-KNN indices and relative xyz for ``[B,3,N]`` points."""
    xyz = xyz_channel_first.permute(0, 2, 1).contiguous()
    return knn_point_with_relative(neighbors, xyz, xyz)


def _pointconv_from_context(
    module: nn.Module,
    points_channel_first: torch.Tensor,
    indices: torch.Tensor,
    relative_xyz: torch.Tensor,
) -> torch.Tensor:
    """Execute PointConv math using an already computed geometric context.

    This follows the official ``PointConv.forward``/``PointConvD.forward``
    operation order while removing repeated KNN and xyz grouping.
    """
    batch_size = int(points_channel_first.shape[0])
    point_count = int(relative_xyz.shape[1])

    points = points_channel_first.permute(0, 2, 1)
    grouped_points = index_points_group(points, indices)
    combined = torch.cat((relative_xyz, grouped_points), dim=-1)

    weights = module.weightnet(relative_xyz.permute(0, 3, 2, 1))
    combined = torch.matmul(
        combined.permute(0, 1, 3, 2),
        weights.permute(0, 3, 2, 1),
    ).reshape(batch_size, point_count, -1)

    combined = module.linear(combined)
    if bool(getattr(module, "bn", False)):
        combined = module.bn_linear(combined.permute(0, 2, 1))
    else:
        combined = combined.permute(0, 2, 1)

    if bool(getattr(module, "use_act", True)):
        combined = module.relu(combined)
    return combined


class EncodedFrame(NamedTuple):
    """Per-frame pyramid reused across adjacent online pairs."""

    points: tuple[torch.Tensor, ...]
    features: tuple[torch.Tensor, ...]
    fps_indices: tuple[torch.Tensor, ...]
    upsample_contexts: tuple[
        tuple[torch.Tensor, torch.Tensor] | None,
        ...,
    ]
    feature_l4_to_l3: torch.Tensor
    # Per-level self 9-NN (indices [B,N,9], relative xyz [B,N,9,3]) for levels
    # 0..3. Source-frame-only geometry used by the GRU/flow estimators; computed
    # once per frame here instead of once per decode (twice at 1024 points).
    self_knn_contexts: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()


def subset_knn_context(
    indices: torch.Tensor,
    relative_xyz: torch.Tensor,
    neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive the exact ``neighbors``-NN from a larger KNN context.

    The k nearest of the 32 nearest are the k nearest overall, so this equals
    ``knn_point_with_relative(neighbors, xyz, xyz)`` (up to distance ties) while
    reading the 393 KB ``relative_xyz`` instead of building a 1024x1024 matrix.
    """
    distances = torch.sum(relative_xyz * relative_xyz, dim=3)
    _, selected = torch.topk(distances, int(neighbors), dim=2, largest=False, sorted=False)
    sub_indices = torch.gather(indices, 2, selected)
    sub_relative = torch.gather(
        relative_xyz, 2, selected.unsqueeze(3).expand(-1, -1, -1, relative_xyz.shape[3])
    )
    return sub_indices, sub_relative


class PointConvEncoder(nn.Module):
    """Original encoder with an exact eval-only level0/level1 KNN reuse path."""

    def __init__(self, weightnet=8):
        super().__init__()
        feat_nei = 32

        self.level0_lift = Conv1d(3, 32)
        self.level0 = PointConv(
            feat_nei,
            32 + 3,
            32,
            weightnet=weightnet,
        )
        self.level0_1 = Conv1d(32, 64)

        self.level1 = PointConvD(
            1024,
            feat_nei,
            64 + 3,
            64,
            weightnet=weightnet,
        )
        self.level1_0 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)

        self.level2 = PointConvD(
            512,
            feat_nei,
            128 + 3,
            128,
            weightnet=weightnet,
        )
        self.level2_0 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)

        self.level3 = PointConvD(
            256,
            feat_nei,
            256 + 3,
            256,
            weightnet=weightnet,
        )
        self.level3_0 = Conv1d(256, 256)
        self.level3_1 = Conv1d(256, 512)

        self.level4 = PointConvD(
            64,
            feat_nei,
            512 + 3,
            256,
            weightnet=weightnet,
        )

        self.register_buffer(
            "_identity_fps_l1",
            torch.arange(
                1024,
                dtype=torch.int32,
            ).unsqueeze(0),
            persistent=False,
        )

    def _identity_fps_indices(
        self,
        batch_size: int,
        point_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        cached = self._identity_fps_l1
        if point_count == cached.shape[1] and cached.device == device:
            if batch_size == 1:
                return cached
            return cached.expand(batch_size, -1).contiguous()

        return (
            torch.arange(
                point_count,
                device=device,
                dtype=torch.int32,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )

    def _forward_eval(
        self,
        xyz: torch.Tensor,
        color: torch.Tensor,
    ):
        """Eval path sharing one 32-NN geometry between levels 0 and 1."""
        indices, relative_xyz = _self_knn_context(
            xyz,
            int(self.level0.nsample),
        )

        feat_l0 = self.level0_lift(color)
        feat_l0 = _pointconv_from_context(
            self.level0,
            feat_l0,
            indices,
            relative_xyz,
        )
        feat_l0_1 = self.level0_1(feat_l0)

        # The checkpoint architecture defines level 1 at exactly 1024 points.
        # Reusing the level-0 neighborhood is exact only when the input already
        # contains 1024 points.  For larger deployment inputs (e.g. 2048), run
        # the original PointConvD downsampling so recurrent1 stays at 1024.
        if xyz.shape[2] == 1024:
            pc_l1 = xyz
            fps_l1 = self._identity_fps_indices(
                xyz.shape[0],
                1024,
                xyz.device,
            )
            feat_l1 = _pointconv_from_context(
                self.level1,
                feat_l0_1,
                indices,
                relative_xyz,
            )
        else:
            pc_l1, feat_l1, fps_l1 = self.level1(
                xyz,
                feat_l0_1,
            )
        feat_l1 = self.level1_0(feat_l1)
        feat_l1_2 = self.level1_1(feat_l1)

        pc_l2, feat_l2, fps_l2 = self.level2(
            pc_l1,
            feat_l1_2,
        )
        feat_l2 = self.level2_0(feat_l2)
        feat_l2_3 = self.level2_1(feat_l2)

        pc_l3, feat_l3, fps_l3 = self.level3(
            pc_l2,
            feat_l2_3,
        )
        feat_l3 = self.level3_0(feat_l3)
        feat_l3_4 = self.level3_1(feat_l3)

        pc_l4, feat_l4, fps_l4 = self.level4(
            pc_l3,
            feat_l3_4,
        )

        self._last_level0_context = (indices, relative_xyz)
        return (
            [xyz, pc_l1, pc_l2, pc_l3, pc_l4],
            [feat_l0, feat_l1, feat_l2, feat_l3, feat_l4],
            [fps_l1, fps_l2, fps_l3, fps_l4],
        )

    def forward_with_context(self, xyz, color):
        """``forward`` plus the level-0 32-NN context (indices, relative xyz)."""
        outputs = self.forward(xyz, color)
        context = self._last_level0_context
        self._last_level0_context = None
        return outputs, context

    def forward(self, xyz, color):
        if self.training:
            raise RuntimeError(
                "This minimal model is inference-only. Call model.eval()."
            )
        if xyz.shape[2] < 1024:
            raise ValueError(
                "The optimized deployment model requires at least "
                f"1024 points, received {xyz.shape[2]}."
            )
        return self._forward_eval(xyz, color)
