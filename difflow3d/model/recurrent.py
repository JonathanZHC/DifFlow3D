"""Recurrent and diffusion flow-refinement blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflow3d.ops.pointnet2 import pointnet2_utils
from .pointconv import (
    BidirectionalLayerFeatCosine,
    Conv1d,
    FlowEmbeddingLayer,
    PointWarping,
    SinusoidalPosEmb,
    cosine_beta_schedule,
    cross_block_apply,
    cross_block_context,
    group_channel_first,
    knn_point,
    knn_point_cosine_prenorm,
    knn_point_with_relative,
    l2_normalize_points,
)

scale = 1.0

class RecurrentUnit(nn.Module):

    def __init__(self, iters, feat_ch, feat_new_ch, latent_ch, cross_mlp1, cross_mlp2, weightnet=8, flow_channels=[64, 64], flow_mlp=[64, 64]):
        super(RecurrentUnit, self).__init__()
        flow_nei = 32
        self.iters = iters
        self.scale = scale
        self.flow_nei = flow_nei
        self.bid = BidirectionalLayerFeatCosine(flow_nei, feat_new_ch + feat_ch, cross_mlp1)
        self.fe = FlowEmbeddingLayer(flow_nei, cross_mlp1[-1], cross_mlp2)
        neighbors = 9
        self.flow = DiffusionSceneFlowGRUResidual(neighbors, in_channel=cross_mlp2[-1] + feat_ch, latent_channel=latent_ch, mlp=flow_channels, channels=flow_channels)
        self.warping = PointWarping()

    def _supports_fast_cross_path(self) -> bool:
        """Check the imported pointconv_util modules expose expected weights."""
        bid_attrs = ('cross_t11', 'cross_t22', 'pos', 'mlp', 'bn', 'relu')
        fe_attrs = ('conv1', 'conv2', 'pos', 'mlp', 'bn', 'relu')
        return all((hasattr(self.bid, name) for name in bid_attrs)) and all((hasattr(self.fe, name) for name in fe_attrs))

    def _prepare_cosine_neighbors(self, feat1, feat2):
        half_neighbors = self.flow_nei // 2
        # Normalise each feature set once; both directions reuse them.
        feat1_n = l2_normalize_points(feat1.permute(0, 2, 1))
        feat2_n = l2_normalize_points(feat2.permute(0, 2, 1))
        cosine_12 = knn_point_cosine_prenorm(half_neighbors, feat2_n, feat1_n)
        cosine_21 = knn_point_cosine_prenorm(half_neighbors, feat1_n, feat2_n)
        return (cosine_12, cosine_21)

    def _prepare_spatial_neighbors(self, pc1, pc2):
        half_neighbors = self.flow_nei // 2
        pc1_n = pc1.permute(0, 2, 1)
        pc2_n = pc2.permute(0, 2, 1)
        spatial_12 = knn_point(half_neighbors, pc2_n, pc1_n)
        spatial_21 = knn_point(half_neighbors, pc1_n, pc2_n)
        return (spatial_12, spatial_21)

    @staticmethod
    def _prepare_combined_cross_context(xyz1, xyz2, cosine_idx, spatial_idx):
        """Combine neighbor indices and cache geometry shared by cross blocks.

        Returns int32 indices and ``[B,3,N1,K]`` offsets in the grouping kernel's
        native layout (see ``pointconv.cross_block_context``).
        """
        combined_idx = torch.cat((cosine_idx, spatial_idx), dim=-1)
        return cross_block_context(xyz1, xyz2, combined_idx)

    @staticmethod
    def _cross_from_combined_context(module, points1, points2, context):
        """Exact cross math using cached indices and geometric offsets."""
        return cross_block_apply(
            points1, points2, module.pos, module.mlp, module.bn, module.relu, context
        )

    def _bidirectional_fast(self, c_feat1, c_feat2, context_12, context_21):
        feat1_new = self._cross_from_combined_context(self.bid, self.bid.cross_t11(c_feat1), self.bid.cross_t22(c_feat2), context_12)
        feat2_new = self._cross_from_combined_context(self.bid, self.bid.cross_t11(c_feat2), self.bid.cross_t22(c_feat1), context_21)
        return (feat1_new, feat2_new)

    def _flow_embedding_fast(self, feat1_new, feat2_new, context_12):
        points1 = self.fe.conv1(feat1_new)
        points2 = self.fe.conv2(feat2_new)
        return self._cross_from_combined_context(self.fe, points1, points2, context_12)

    def forward(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_flow, up_feat, gt_flow=None, certainty=None, uncertainty=0.5, self_knn_context=None):
        c_feat1 = torch.cat([feat1, feat1_new], dim=1)
        c_feat2 = torch.cat([feat2, feat2_new], dim=1)
        flows = []
        use_fast_cross = not self.training and self._supports_fast_cross_path()
        if use_fast_cross:
            cosine_12, cosine_21 = self._prepare_cosine_neighbors(feat1, feat2)
            if self_knn_context is not None:
                # Source-frame-only geometry precomputed in encode_frame.
                flow_neighbor_context = self.flow._neighbor_context_from_knn(*self_knn_context)
            else:
                flow_neighbor_context = self.flow._prepare_neighbor_context(pc1, pc1)
            flow_time_per_point = self.flow._prepare_eval_time_per_point(up_flow)
        else:
            cosine_12 = cosine_21 = None
            flow_neighbor_context = None
            flow_time_per_point = None
        for iteration in range(self.iters):
            pc2_warp = self.warping(pc1, pc2, up_flow)
            if use_fast_cross:
                spatial_12, spatial_21 = self._prepare_spatial_neighbors(pc1, pc2_warp)
                context_12 = self._prepare_combined_cross_context(pc1, pc2_warp, cosine_12, spatial_12)
                context_21 = self._prepare_combined_cross_context(pc2_warp, pc1, cosine_21, spatial_21)
                feat1_new, feat2_new = self._bidirectional_fast(c_feat1, c_feat2, context_12, context_21)
                fe = self._flow_embedding_fast(feat1_new, feat2_new, context_12)
            else:
                feat1_new, feat2_new = self.bid(pc1, pc2_warp, c_feat1, c_feat2, feat1, feat2)
                fe = self.fe(pc1, pc2_warp, feat1_new, feat2_new, feat1, feat2)
            new_feat1 = torch.cat([feat1, fe], dim=1)
            if self.training:
                feat_flow, flow, certainty_new, loss = self.flow(pc1, pc1, up_feat, new_feat1, up_flow, gt_flow, certainty, uncertainty)
            elif use_fast_cross:
                feat_flow, flow, certainty_new = self.flow._forward_eval_fast(pc1, pc1, up_feat, new_feat1, up_flow, gt_flow, certainty, uncertainty, neighbor_context=flow_neighbor_context, time_per_point=flow_time_per_point)
            else:
                feat_flow, flow, certainty_new = self.flow(pc1, pc1, up_feat, new_feat1, up_flow, gt_flow, certainty, uncertainty)
            up_flow = flow
            up_feat = feat_flow
            flows.append(flow)
            if iteration + 1 < self.iters:
                c_feat1 = torch.cat([feat1, feat1_new], dim=1)
                c_feat2 = torch.cat([feat2, feat2_new], dim=1)
        if self.training:
            return (flows, feat1_new, feat2_new, feat_flow, certainty_new, loss)
        return (flows, feat1_new, feat2_new, feat_flow, certainty_new)




class DiffusionSceneFlowGRUResidual(nn.Module):

    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn=False, use_leaky=True, return_inter=False, radius=None, use_relu=False, channels=[64, 64], clamp=[-200, 200], scale_dif=1.0):
        super(DiffusionSceneFlowGRUResidual, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.use_relu = use_relu
        self.fc = nn.Conv1d(channels[-1], 4, 1)
        self.clamp = clamp
        last_channel = in_channel + 3 + 64 + 3 + 1
        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)
        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)
        timesteps = 1000
        sampling_timesteps = 1
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps=timesteps).float()
        self.sampling_timesteps = sampling_timesteps
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.01
        self.scale = scale_dif
        self.snr_scale = self.scale
        time_dim = 64
        dim = 16
        sinu_pos_emb = SinusoidalPosEmb(dim)
        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', log_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.iters = 1
        self._eval_time_cache = {}

    def train(self, mode: bool=True):
        if mode:
            self._eval_time_cache.clear()
        return super().train(mode)

    def _apply(self, fn):
        self._eval_time_cache.clear()
        return super()._apply(fn)

    @staticmethod
    def _neighbor_context_from_knn(knn_idx, relative_xyz):
        """``([B,N,K] idx, [B,N,K,3] rel)`` -> (int32 idx, ``[B,3,N,K]`` rel)."""
        return (
            knn_idx.int().contiguous(),
            relative_xyz.permute(0, 3, 1, 2).contiguous(),
        )

    def _prepare_neighbor_context(self, xyz1, xyz2):
        """Prepare geometry-only self-neighborhood data reusable across calls."""
        xyz1_n = xyz1.permute(0, 2, 1).contiguous()
        xyz2_n = xyz2.permute(0, 2, 1).contiguous()
        knn_idx, direction_xyz = knn_point_with_relative(
            self.nsample, xyz2_n, xyz1_n
        )
        return self._neighbor_context_from_knn(knn_idx, direction_xyz)

    def _prepare_eval_time_per_point(self, flow):
        """Return cached deterministic one-step DDIM time features.

        During evaluation the timestep is always T-1 and the module weights
        do not change.  Reusing this tensor removes the time MLP, allocation,
        and expansion launches from every subsequent frame and also shrinks
        the captured CUDA graph.
        """
        batch_size = int(flow.shape[0])
        point_count = int(flow.shape[2])
        cache_key = (batch_size, point_count, flow.device, flow.dtype)
        cached = self._eval_time_cache.get(cache_key)
        if cached is not None:
            return cached
        with torch.no_grad():
            t = torch.full((batch_size,), self.timesteps - 1, device=flow.device, dtype=torch.long)
            # Channel-first [B,64,N] so it broadcasts along K in the GRU's
            # [B,C,N,K] layout without a transpose.
            cached = self.time_mlp(t).unsqueeze(2).expand(-1, -1, point_count).contiguous().detach()
        self._eval_time_cache[cache_key] = cached
        return cached

    def _gru_update_eval(self, points1, points2, delta_flow, delta_certainty, time_per_point, neighbor_context):
        # Everything below is in [B,C,N,K] (grouping-native) layout: one
        # contiguous concat feeds the r/z/h conv chains directly instead of a
        # permuted view that aten had to materialise three times.
        knn_idx_int, direction_xyz = neighbor_context
        nsample = self.nsample
        grouped_points2 = group_channel_first(points2, knn_idx_int)
        time_grouped = time_per_point.unsqueeze(3).expand(-1, -1, -1, nsample)
        delta_flow_grouped = delta_flow.unsqueeze(3).expand(-1, -1, -1, nsample)
        delta_certainty_grouped = delta_certainty.unsqueeze(3).expand(-1, -1, -1, nsample)
        new_points = torch.cat([grouped_points2, direction_xyz, delta_certainty_grouped, delta_flow_grouped, time_grouped], dim=1)
        point1_graph = points1
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                r = r + self.fuse_r(point1_graph).unsqueeze(3)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
            else:
                r = self.relu(r)
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                z = z + self.fuse_z(point1_graph).unsqueeze(3)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
            else:
                z = self.relu(z)
            if i == len(self.mlp_z_convs) - 2:
                z = z.amax(dim=3, keepdim=True)
        z = z.squeeze(3)
        point1_expand = self.fuse_r_o(r * point1_graph.unsqueeze(3))
        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                h = self.relu(h) if self.use_relu else self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = h.amax(dim=3, keepdim=True)
        h = h.squeeze(3)

        # Keep recurrent-state tensors dtype-consistent at this boundary.
        # These checks are no-ops in the FP32 deployment path.
        state_dtype = points1.dtype
        if h.dtype != state_dtype:
            h = h.to(dtype=state_dtype)
        if z.dtype != state_dtype:
            z = z.to(dtype=state_dtype)
        new_points = torch.lerp(points1, h, z)

        if self.mlp2:
            for conv in self.mlp2:
                new_points = conv(new_points)

        update = self.fc(new_points - points1).float()
        delta_flow = update[:, :3, :].clamp(self.clamp[0], self.clamp[1])
        delta_certainty = update[:, 3:, :]
        return (new_points, delta_flow, delta_certainty)

    def _forward_eval_fast(self, xyz1, xyz2, points1, points2, flow, flow_gt, certainty, uncertainty=0.5, neighbor_context=None, time_per_point=None):
        """Equivalent fast path for the configured one-step DDIM inference."""
        del flow_gt, uncertainty
        if neighbor_context is None:
            neighbor_context = self._prepare_neighbor_context(xyz1, xyz2)
        if time_per_point is None:
            time_per_point = self._prepare_eval_time_per_point(flow)
        delta_flow = (self.scale * torch.randn_like(flow)).float()
        delta_certainty = (self.scale * torch.randn_like(certainty)).float()
        new_points = points1
        for _ in range(self.iters):
            new_points, delta_flow, delta_certainty = self._gru_update_eval(points1, points2, delta_flow.detach(), delta_certainty.detach(), time_per_point, neighbor_context)
        flow_new = delta_flow if flow is None else delta_flow + flow
        certainty_new = certainty + delta_certainty
        return (new_points, flow_new, certainty_new)


    def forward(
        self,
        xyz1,
        xyz2,
        points1,
        points2,
        flow,
        flow_gt,
        certainty,
        uncertainty=0.5,
    ):
        if self.training:
            raise RuntimeError(
                "This minimal model is inference-only. Call model.eval()."
            )
        return self._forward_eval_fast(
            xyz1,
            xyz2,
            points1,
            points2,
            flow,
            flow_gt,
            certainty,
            uncertainty,
        )
