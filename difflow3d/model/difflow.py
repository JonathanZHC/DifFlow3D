"""Checkpoint-compatible DifFlow3D network definition."""

import torch
import torch.nn as nn

from .encoder import EncodedFrame, PointConvEncoder, _pointconv_from_context, _self_knn_context
from .recurrent import RecurrentUnit
from .pointconv import (
    Conv1d,
    CrossLayerLightFeatCosine as CrossLayer,
    SceneFlowEstimatorResidual,
    index_points_group,
    knn_point_with_relative,
)

scale = 1.0

class PointConvBidirection(nn.Module):
    """DifFlow3D with split encode/decode APIs for streaming reuse."""

    def __init__(
        self,
        iters=3,
        *,
        coarse_iters: int | None = None,
        middle_iters: int | None = None,
        fine_iters: int | None = None,
    ):
        super().__init__()
        flow_nei = 32
        weightnet = 8

        legacy_iters = int(iters)
        coarse_iters = legacy_iters if coarse_iters is None else int(coarse_iters)
        middle_iters = legacy_iters if middle_iters is None else int(middle_iters)
        fine_iters = legacy_iters if fine_iters is None else int(fine_iters)
        if min(coarse_iters, middle_iters, fine_iters) < 1:
            raise ValueError("All recurrent iteration counts must be >= 1.")

        self.scale = scale
        # ``iters`` is kept for backward compatibility with callers that inspect
        # the legacy scalar value. Runtime code uses the explicit per-level map.
        self.iters = legacy_iters
        self.coarse_iters = coarse_iters
        self.middle_iters = middle_iters
        self.fine_iters = fine_iters
        self.iterations = {
            "coarse": coarse_iters,
            "middle": middle_iters,
            "fine": fine_iters,
        }

        self.encoder = PointConvEncoder(weightnet=weightnet)

        # recurrent0/1/2 operate at fine/middle/coarse resolution respectively.
        self.recurrent0 = RecurrentUnit(
            iters=fine_iters,
            feat_ch=32,
            feat_new_ch=32,
            latent_ch=64,
            cross_mlp1=[32, 32],
            cross_mlp2=[32, 32],
            weightnet=weightnet,
            flow_channels=[64, 64],
            flow_mlp=[64, 64],
        )
        self.recurrent1 = RecurrentUnit(
            iters=middle_iters,
            feat_ch=64,
            feat_new_ch=64,
            latent_ch=64,
            cross_mlp1=[64, 64],
            cross_mlp2=[64, 64],
            weightnet=weightnet,
        )
        self.recurrent2 = RecurrentUnit(
            iters=coarse_iters,
            feat_ch=128,
            feat_new_ch=128,
            latent_ch=64,
            cross_mlp1=[128, 128],
            cross_mlp2=[128, 128],
            weightnet=weightnet,
        )

        self.cross3 = CrossLayer(
            flow_nei,
            256 + 64,
            [256, 256],
            [256, 256],
        )
        self.flow3 = SceneFlowEstimatorResidual(
            256,
            256,
            channels=[128, 64],
            mlp=[],
            weightnet=weightnet,
        )

        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 128)
        self.deconv2_1 = Conv1d(128, 64)
        self.deconv1_0 = Conv1d(64, 32)


    @staticmethod
    def _prepare_upsample_context(
        xyz: torch.Tensor,
        sparse_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xyz_n = xyz.permute(0, 2, 1)
        sparse_xyz_n = sparse_xyz.permute(0, 2, 1)

        indices, relative = knn_point_with_relative(
            3, sparse_xyz_n.contiguous(), xyz_n.contiguous()
        )
        distance = torch.sqrt(
            torch.sum(relative * relative, dim=3)
        ).clamp_min(1.0e-10)
        inverse = distance.reciprocal()
        weights = inverse / inverse.sum(dim=2, keepdim=True)
        return indices, weights

    @staticmethod
    def _apply_upsample_context(
        sparse_values: torch.Tensor,
        context: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        indices, weights = context
        grouped = index_points_group(
            sparse_values.permute(0, 2, 1),
            indices,
        )
        dense = torch.sum(
            weights.unsqueeze(-1) * grouped,
            dim=2,
        )
        return dense.permute(0, 2, 1)

    def encode_frame(
        self,
        xyz: torch.Tensor,
        color: torch.Tensor | None = None,
    ) -> EncodedFrame:
        """Encode one ``[B,N,3]`` frame once for adjacent-pair reuse."""
        if self.training:
            raise RuntimeError("encode_frame is an evaluation-only API.")
        if color is None:
            color = xyz

        xyz_cf = xyz.permute(0, 2, 1)
        color_cf = color.permute(0, 2, 1)
        points, features, indices = self.encoder(
            xyz_cf,
            color_cf,
        )

        context_43 = self._prepare_upsample_context(
            points[3],
            points[4],
        )
        context_32 = self._prepare_upsample_context(
            points[2],
            points[3],
        )
        context_21 = self._prepare_upsample_context(
            points[1],
            points[2],
        )
        context_10 = self._prepare_upsample_context(
            points[0],
            points[1],
        )

        feature_l4_to_l3 = self.deconv4_3(
            self._apply_upsample_context(
                features[4],
                context_43,
            )
        )

        return EncodedFrame(
            points=tuple(points),
            features=tuple(features),
            fps_indices=tuple(indices),
            upsample_contexts=(
                context_43,
                context_32,
                context_21,
                context_10,
            ),
            feature_l4_to_l3=feature_l4_to_l3,
        )

    def _flow3_eval(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        cost_volume: torch.Tensor,
    ):
        """SceneFlowEstimatorResidual with one shared self-KNN context."""
        pointconvs = getattr(self.flow3, "pointconv_list", None)
        if not pointconvs:
            return self.flow3(xyz, features, cost_volume)

        first = pointconvs[0]
        neighbors = int(getattr(first, "nsample", 9))
        indices, relative_xyz = _self_knn_context(
            xyz,
            neighbors,
        )

        new_points = torch.cat(
            (features, cost_volume),
            dim=1,
        )
        for pointconv in pointconvs:
            if int(getattr(pointconv, "nsample", -1)) != neighbors:
                return self.flow3(xyz, features, cost_volume)
            new_points = _pointconv_from_context(
                pointconv,
                new_points,
                indices,
                relative_xyz,
            )

        for conv in self.flow3.mlp_convs:
            new_points = conv(new_points)

        update = self.flow3.fc(new_points)
        flow = update[:, :3, :].clamp(
            self.flow3.clamp[0],
            self.flow3.clamp[1],
        )
        certainty = update[:, 3:, :]
        return new_points, flow, certainty

    def decode_pair(
        self,
        source: EncodedFrame,
        target: EncodedFrame,
        gt_flow: torch.Tensor | None = None,
        uncertainty: float = 0.5,
    ):
        """Decode a source/target encoded pair without re-running encoders."""
        if self.training:
            raise RuntimeError("decode_pair is an evaluation-only API.")
        del gt_flow

        pc1s = source.points
        pc2s = target.points
        feat1s = source.features
        feat2s = target.features
        idx1s = source.fps_indices
        idx2s = target.fps_indices

        _, source_32, source_21, source_10 = (
            source.upsample_contexts
        )
        _, target_32, target_21, target_10 = (
            target.upsample_contexts
        )

        c_feat1_l3 = torch.cat(
            (feat1s[3], source.feature_l4_to_l3),
            dim=1,
        )
        c_feat2_l3 = torch.cat(
            (feat2s[3], target.feature_l4_to_l3),
            dim=1,
        )

        (
            feat1_new_l3,
            feat2_new_l3,
            cross3,
        ) = self.cross3(
            pc1s[3],
            pc2s[3],
            c_feat1_l3,
            c_feat2_l3,
            feat1s[3],
            feat2s[3],
        )

        feat3, flow3, certainty3 = self._flow3_eval(
            pc1s[3],
            feat1s[3],
            cross3,
        )

        feat1_l3_2 = self.deconv3_2(
            self._apply_upsample_context(
                feat1_new_l3,
                source_32,
            )
        )
        feat2_l3_2 = self.deconv3_2(
            self._apply_upsample_context(
                feat2_new_l3,
                target_32,
            )
        )

        up_flow2 = self._apply_upsample_context(
            self.scale * flow3,
            source_32,
        )
        up_certainty2 = self._apply_upsample_context(
            self.scale * certainty3,
            source_32,
        )
        up_feat2 = self._apply_upsample_context(
            feat3,
            source_32,
        )

        (
            flows2,
            feat1_new_l2,
            feat2_new_l2,
            feat2,
            certainty2,
        ) = self.recurrent2(
            pc1s[2],
            pc2s[2],
            feat1_l3_2,
            feat2_l3_2,
            feat1s[2],
            feat2s[2],
            up_flow2,
            up_feat2,
            None,
            up_certainty2,
            uncertainty,
        )

        feat1_l2_1 = self.deconv2_1(
            self._apply_upsample_context(
                feat1_new_l2,
                source_21,
            )
        )
        feat2_l2_1 = self.deconv2_1(
            self._apply_upsample_context(
                feat2_new_l2,
                target_21,
            )
        )

        up_flow1 = self._apply_upsample_context(
            self.scale * flows2[-1],
            source_21,
        )
        up_certainty1 = self._apply_upsample_context(
            self.scale * certainty2,
            source_21,
        )
        up_feat1 = self._apply_upsample_context(
            feat2,
            source_21,
        )

        (
            flows1,
            feat1_new_l1,
            feat2_new_l1,
            feat1,
            certainty1,
        ) = self.recurrent1(
            pc1s[1],
            pc2s[1],
            feat1_l2_1,
            feat2_l2_1,
            feat1s[1],
            feat2s[1],
            up_flow1,
            up_feat1,
            None,
            up_certainty1,
            uncertainty,
        )

        feat1_l1_0 = self.deconv1_0(
            self._apply_upsample_context(
                feat1_new_l1,
                source_10,
            )
        )
        feat2_l1_0 = self.deconv1_0(
            self._apply_upsample_context(
                feat2_new_l1,
                target_10,
            )
        )

        up_flow0 = self._apply_upsample_context(
            self.scale * flows1[-1],
            source_10,
        )
        up_certainty0 = self._apply_upsample_context(
            self.scale * certainty1,
            source_10,
        )
        up_feat0 = self._apply_upsample_context(
            feat1,
            source_10,
        )

        (
            flows0,
            feat1_new_l0,
            feat2_new_l0,
            feat0,
            certainty0,
        ) = self.recurrent0(
            pc1s[0],
            pc2s[0],
            feat1_l1_0,
            feat2_l1_0,
            feat1s[0],
            feat2s[0],
            up_flow0,
            up_feat0,
            None,
            up_certainty0,
            uncertainty,
        )

        flows = [
            flows0[::-1],
            flows1[::-1],
            flows2[::-1],
            [flow3],
        ]
        fps_pc1_idxs = [
            [None for _ in range(max(len(flows0) - 1, 0))],
            [idx1s[0]],
            [idx1s[1]],
            [idx1s[2]],
        ]
        fps_pc2_idxs = [
            [None for _ in range(max(len(flows0) - 1, 0))],
            [idx2s[0]],
            [idx2s[1]],
            [idx2s[2]],
        ]
        return (
            flows,
            fps_pc1_idxs,
            fps_pc2_idxs,
            list(pc1s),
            list(pc2s),
        )





    def forward(
        self,
        xyz1,
        xyz2,
        color1,
        color2,
        gt_flow,
        uncertainty=0.5,
    ):
        del gt_flow
        if self.training:
            raise RuntimeError(
                "This minimal model is inference-only. Call model.eval()."
            )

        source = self.encode_frame(xyz1, color1)
        target = self.encode_frame(xyz2, color2)
        return self.decode_pair(
            source,
            target,
            None,
            uncertainty,
        )
