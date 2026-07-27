"""Production DifFlow3D implementation with evaluation-only acceleration.

The network architecture, trainable parameter shapes, public forward signatures,
checkpoint keys, and return structures remain compatible with the original model.
The evaluation path reuses deterministic KNN/interpolation contexts, combines
neighbor grouping, caches the fixed one-step diffusion time embedding, skips
redundant full-set FPS, and removes dead one-step DDIM work.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, CrossLayerLightFeatCosine as CrossLayer, FlowEmbeddingLayer, BidirectionalLayerFeatCosine
from pointconv_util import SceneFlowEstimatorResidual
from pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance, knn_point_cosine, knn_point
from pointconv_util import default, extract, cosine_beta_schedule, SinusoidalPosEmb
from pointconv_util import index_points_gather


def configure_fast_inference(enable_tf32: bool = True) -> None:
    """Configure global CUDA math settings used by the fast inference path.

    TF32 was beneficial when combined with CUDA Graph replay on RTX 5090.
    Disable it for strict FP32 comparison.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
    torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
    torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")


scale = 1.0

class PointConvEncoder(nn.Module):

    def __init__(self, weightnet=8):
        super(PointConvEncoder, self).__init__()
        feat_nei = 32
        self.level0_lift = Conv1d(3, 32)
        self.level0 = PointConv(feat_nei, 32 + 3, 32, weightnet=weightnet)
        self.level0_1 = Conv1d(32, 64)
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64, weightnet=weightnet)
        self.level1_0 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)
        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128, weightnet=weightnet)
        self.level2_0 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)
        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256, weightnet=weightnet)
        self.level3_0 = Conv1d(256, 256)
        self.level3_1 = Conv1d(256, 512)
        self.level4 = PointConvD(64, feat_nei, 512 + 3, 256, weightnet=weightnet)
        self.register_buffer('_identity_fps_l1', torch.arange(2048, dtype=torch.int32).unsqueeze(0), persistent=False)

    def _identity_fps_indices(self, batch_size: int, point_count: int, device: torch.device) -> torch.Tensor:
        """Return contiguous identity indices accepted by gather_operation."""
        cached = self._identity_fps_l1
        if point_count == cached.shape[1] and cached.device == device:
            if batch_size == 1:
                return cached
            return cached.repeat(batch_size, 1).contiguous()
        return torch.arange(point_count, device=device, dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1).contiguous()

    def forward(self, xyz, color):
        feat_l0 = self.level0_lift(color)
        feat_l0 = self.level0(xyz, feat_l0)
        feat_l0_1 = self.level0_1(feat_l0)
        level1_npoint = getattr(self.level1, 'npoint', None)
        if not self.training and level1_npoint is not None and (xyz.shape[2] == int(level1_npoint)):
            fps_l1 = self._identity_fps_indices(xyz.shape[0], xyz.shape[2], xyz.device)
            pc_l1, feat_l1, fps_l1 = self.level1(xyz, feat_l0_1, fps_idx=fps_l1)
        else:
            pc_l1, feat_l1, fps_l1 = self.level1(xyz, feat_l0_1)
        feat_l1 = self.level1_0(feat_l1)
        feat_l1_2 = self.level1_1(feat_l1)
        pc_l2, feat_l2, fps_l2 = self.level2(pc_l1, feat_l1_2)
        feat_l2 = self.level2_0(feat_l2)
        feat_l2_3 = self.level2_1(feat_l2)
        pc_l3, feat_l3, fps_l3 = self.level3(pc_l2, feat_l2_3)
        feat_l3 = self.level3_0(feat_l3)
        feat_l3_4 = self.level3_1(feat_l3)
        pc_l4, feat_l4, fps_l4 = self.level4(pc_l3, feat_l3_4)
        return ([xyz, pc_l1, pc_l2, pc_l3, pc_l4], [feat_l0, feat_l1, feat_l2, feat_l3, feat_l4], [fps_l1, fps_l2, fps_l3, fps_l4])

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
        feat1_n = feat1.permute(0, 2, 1)
        feat2_n = feat2.permute(0, 2, 1)
        cosine_12 = knn_point_cosine(half_neighbors, feat2_n, feat1_n)
        cosine_21 = knn_point_cosine(half_neighbors, feat1_n, feat2_n)
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
        """Combine neighbor indices and cache geometry shared by cross blocks."""
        combined_idx = torch.cat((cosine_idx, spatial_idx), dim=-1)
        xyz1_n = xyz1.permute(0, 2, 1)
        xyz2_n = xyz2.permute(0, 2, 1)
        neighbor_xyz = index_points_group(xyz2_n, combined_idx)
        direction_xyz = neighbor_xyz - xyz1_n.unsqueeze(2)
        return (combined_idx, direction_xyz)

    @staticmethod
    def _cross_from_combined_context(module, points1, points2, combined_idx, direction_xyz):
        """Exact cross math using cached indices and geometric offsets."""
        points1_n = points1.permute(0, 2, 1)
        points2_n = points2.permute(0, 2, 1)
        neighbor_count = combined_idx.shape[-1]
        grouped_points2 = index_points_group(points2_n, combined_idx).permute(0, 3, 2, 1)
        grouped_points1 = points1_n.unsqueeze(2).expand(-1, -1, neighbor_count, -1).permute(0, 3, 2, 1)
        direction_features = module.pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = module.relu(module.bn(grouped_points2 + grouped_points1 + direction_features))
        for conv in module.mlp:
            new_points = conv(new_points)
        return F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

    def _bidirectional_fast(self, c_feat1, c_feat2, context_12, context_21):
        combined_12, direction_12 = context_12
        combined_21, direction_21 = context_21
        feat1_new = self._cross_from_combined_context(self.bid, self.bid.cross_t11(c_feat1), self.bid.cross_t22(c_feat2), combined_12, direction_12)
        feat2_new = self._cross_from_combined_context(self.bid, self.bid.cross_t11(c_feat2), self.bid.cross_t22(c_feat1), combined_21, direction_21)
        return (feat1_new, feat2_new)

    def _flow_embedding_fast(self, feat1_new, feat2_new, context_12):
        combined_12, direction_12 = context_12
        points1 = self.fe.conv1(feat1_new)
        points2 = self.fe.conv2(feat2_new)
        return self._cross_from_combined_context(self.fe, points1, points2, combined_12, direction_12)

    def forward(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_flow, up_feat, gt_flow=None, certainty=None, uncertainty=0.5):
        c_feat1 = torch.cat([feat1, feat1_new], dim=1)
        c_feat2 = torch.cat([feat2, feat2_new], dim=1)
        flows = []
        use_fast_cross = not self.training and self._supports_fast_cross_path()
        if use_fast_cross:
            cosine_12, cosine_21 = self._prepare_cosine_neighbors(feat1, feat2)
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

class GRUMappingNoGCN(nn.Module):

    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn=False, use_leaky=True, return_inter=False, radius=None, use_relu=False):
        super(GRUMappingNoGCN, self).__init__()
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
        last_channel = in_channel + 3
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

    def forward(self, xyz1, xyz2, points1, points2, flow=None, flow_gt=None):
        B, C, N1 = xyz1.shape
        xyz1_n = xyz1.permute(0, 2, 1)
        xyz2_n = xyz2.permute(0, 2, 1)
        if self.radius is None:
            sqrdists = square_distance(xyz1_n, xyz2_n)
            _, knn_idx = torch.topk(sqrdists, self.nsample, dim=-1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2_n, knn_idx)
            direction_xyz = neighbor_xyz - xyz1_n.unsqueeze(2)
            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx)
            new_points = torch.cat([grouped_points2, direction_xyz], dim=-1).permute(0, 3, 2, 1)
        else:
            new_points = self.queryandgroup(xyz2_n.contiguous(), xyz1_n.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)
        point1_graph = points1
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                r = r + self.fuse_r(point1_graph).unsqueeze(2)
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
                z = z + self.fuse_z(point1_graph).unsqueeze(2)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
            else:
                z = self.relu(z)
            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
        z = z.squeeze(-2)
        point1_expand = r * point1_graph.unsqueeze(2)
        point1_expand = self.fuse_r_o(point1_expand)
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
                h = torch.max(h, -2)[0].unsqueeze(-2)
        h = h.squeeze(-2)
        new_points = (1 - z) * points1 + z * h
        if self.mlp2:
            for conv in self.mlp2:
                new_points = conv(new_points)
        return new_points

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
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
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

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.scale * torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prepare_neighbor_context(self, xyz1, xyz2):
        """Prepare geometry-only self-neighborhood data reusable across calls."""
        batch_size, coordinate_channels, point_count = xyz1.shape
        xyz1_n = xyz1.permute(0, 2, 1)
        xyz2_n = xyz2.permute(0, 2, 1)
        sqrdists = square_distance(xyz1_n, xyz2_n)
        _, knn_idx = torch.topk(sqrdists, self.nsample, dim=-1, largest=False, sorted=False)
        neighbor_xyz = index_points_group(xyz2_n, knn_idx)
        direction_xyz = neighbor_xyz - xyz1_n.unsqueeze(2)
        return (knn_idx, direction_xyz)

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
            cached = self.time_mlp(t).unsqueeze(1).expand(-1, point_count, -1).detach()
        self._eval_time_cache[cache_key] = cached
        return cached

    def _gru_update_eval(self, points1, points2, delta_flow, delta_certainty, time_per_point, neighbor_context):
        knn_idx, direction_xyz = neighbor_context
        batch_size, _, point_count = points1.shape
        grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx)
        time_grouped = time_per_point.unsqueeze(2).expand(-1, -1, self.nsample, -1)
        delta_flow_grouped = delta_flow.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, self.nsample, -1)
        delta_certainty_grouped = delta_certainty.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, self.nsample, -1)
        new_points = torch.cat([grouped_points2, direction_xyz, delta_certainty_grouped, delta_flow_grouped, time_grouped], dim=-1).permute(0, 3, 2, 1)
        point1_graph = points1
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                r = r + self.fuse_r(point1_graph).unsqueeze(2)
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
                z = z + self.fuse_z(point1_graph).unsqueeze(2)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
            else:
                z = self.relu(z)
            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
        z = z.squeeze(-2)
        point1_expand = self.fuse_r_o(r * point1_graph.unsqueeze(2))
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
                h = torch.max(h, -2)[0].unsqueeze(-2)
        h = h.squeeze(-2)
        new_points = (1 - z) * points1 + z * h
        if self.mlp2:
            for conv in self.mlp2:
                new_points = conv(new_points)
        update = self.fc(new_points - points1)
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

    def forward(self, xyz1, xyz2, points1, points2, flow, flow_gt, certainty, uncertainty=0.5):
        if not self.training:
            return self._forward_eval_fast(xyz1, xyz2, points1, points2, flow, flow_gt, certainty, uncertainty)
        B, C, N1 = xyz1.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        batch_size = flow.shape[0]
        n = flow.shape[2]
        flow_gt = flow_gt.permute(0, 2, 1)
        certainty = certainty.permute(0, 2, 1)
        gt_certainty_norm = torch.norm(flow_gt - flow, dim=-2)
        sf_norm = torch.norm(flow_gt, dim=-2)
        relative_err = gt_certainty_norm / (sf_norm + 0.0001)
        gt_certainty = torch.where(torch.logical_or(gt_certainty_norm < uncertainty, relative_err < uncertainty), torch.ones_like(gt_certainty_norm), torch.zeros_like(gt_certainty_norm))
        gt_certainty = torch.unsqueeze(gt_certainty, dim=2)
        gt_delta_certainty = (gt_certainty - certainty).detach()
        gt_delta_flow = flow_gt - flow
        gt_delta_flow = torch.where(torch.isinf(gt_delta_flow), torch.zeros_like(gt_delta_flow), gt_delta_flow).detach()
        t = torch.randint(0, self.timesteps, (batch_size,), device=flow.device).long()
        noise = (self.scale * torch.randn_like(gt_delta_flow)).float()
        noise_certainty = (self.scale * torch.randn_like(gt_delta_certainty)).float()
        delta_flow = self.q_sample(gt_delta_flow, t, noise=noise)
        flow_new = flow + delta_flow
        delta_certainty = self.q_sample(gt_delta_certainty, t, noise=noise_certainty)
        certainty_new = certainty + delta_certainty
        for _ in range(self.iters):
            delta_flow = delta_flow.detach()
            flow_new = flow_new.detach()
            delta_certainty = delta_certainty.detach()
            certainty_new = certainty_new.detach()
            time_features = self.time_mlp(t)
            time_features = time_features.unsqueeze(1).repeat(1, n, 1)
            if self.radius is None:
                sqrdists = square_distance(xyz1, xyz2)
                _, knn_idx = torch.topk(sqrdists, self.nsample, dim=-1, largest=False, sorted=False)
                neighbor_xyz = index_points_group(xyz2, knn_idx)
                direction_xyz = neighbor_xyz - xyz1.unsqueeze(2)
                grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx)
                time_grouped = time_features.unsqueeze(-2).repeat(1, 1, self.nsample, 1)
                delta_flow_grouped = delta_flow.permute(0, 2, 1).unsqueeze(-2).repeat(1, 1, self.nsample, 1)
                delta_certainty_grouped = delta_certainty.unsqueeze(-2).repeat(1, 1, self.nsample, 1)
                new_points = torch.cat([grouped_points2, direction_xyz, delta_certainty_grouped, delta_flow_grouped, time_grouped], dim=-1).permute(0, 3, 2, 1)
            else:
                new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
                new_points = new_points.permute(0, 1, 3, 2)
            point1_graph = points1
            r = new_points
            for i, conv in enumerate(self.mlp_r_convs):
                r = conv(r)
                if i == 0:
                    r = r + self.fuse_r(point1_graph).unsqueeze(2)
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
                    z = z + self.fuse_z(point1_graph).unsqueeze(2)
                if self.bn:
                    z = self.mlp_z_bns[i](z)
                if i == len(self.mlp_z_convs) - 1:
                    z = self.sigmoid(z)
                else:
                    z = self.relu(z)
                if i == len(self.mlp_z_convs) - 2:
                    z = torch.max(z, -2)[0].unsqueeze(-2)
            z = z.squeeze(-2)
            point1_expand = self.fuse_r_o(r * point1_graph.unsqueeze(2))
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
                    h = torch.max(h, -2)[0].unsqueeze(-2)
            h = h.squeeze(-2)
            new_points = (1 - z) * points1 + z * h
            if self.mlp2:
                for conv in self.mlp2:
                    new_points = conv(new_points)
            update = self.fc(new_points - points1)
            delta_flow = update[:, :3, :].clamp(self.clamp[0], self.clamp[1])
            delta_certainty = update[:, 3:, :]
            certainty = certainty.permute(0, 2, 1)
            certainty_new = certainty + delta_certainty
            flow_new = delta_flow if flow is None else delta_flow + flow
            loss_df = F.mse_loss(delta_flow, gt_delta_flow)
            gt_delta_certainty = gt_delta_certainty.permute(0, 2, 1)
            loss_dc = F.mse_loss(delta_certainty, gt_delta_certainty)
            loss = loss_df + loss_dc
        return (new_points, flow_new, certainty_new, loss)

class SceneFlowGRUResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch=3, channels=[64, 64], mlp=[64, 64], neighbors=9, clamp=[-200, 200], use_leaky=True):
        super(SceneFlowGRUResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        self.gru = GRUMappingNoGCN(neighbors, in_channel=cost_ch, latent_channel=feat_ch, mlp=channels)
        self.fc = nn.Conv1d(channels[-1], 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None, flow_gt=None):
        """
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        """
        feats_new = self.gru(xyz, xyz, feats, cost_volume, flow, flow_gt)
        new_points = feats_new - feats
        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1])
        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return (feats_new, flow)

class PointConvBidirection(nn.Module):

    def __init__(self, iters=3):
        super(PointConvBidirection, self).__init__()
        flow_nei = 32
        weightnet = 8
        self.scale = scale
        self.iters = iters
        self.encoder = PointConvEncoder(weightnet=weightnet)
        self.recurrent0 = RecurrentUnit(iters=iters, feat_ch=32, feat_new_ch=32, latent_ch=64, cross_mlp1=[32, 32], cross_mlp2=[32, 32], weightnet=weightnet, flow_channels=[64, 64], flow_mlp=[64, 64])
        self.recurrent1 = RecurrentUnit(iters=iters, feat_ch=64, feat_new_ch=64, latent_ch=64, cross_mlp1=[64, 64], cross_mlp2=[64, 64], weightnet=weightnet)
        self.recurrent2 = RecurrentUnit(iters=iters, feat_ch=128, feat_new_ch=128, latent_ch=64, cross_mlp1=[128, 128], cross_mlp2=[128, 128], weightnet=weightnet)
        self.cross3 = CrossLayer(flow_nei, 256 + 64, [256, 256], [256, 256])
        self.flow3 = SceneFlowEstimatorResidual(256, 256, channels=[128, 64], mlp=[], weightnet=weightnet)
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 128)
        self.deconv2_1 = Conv1d(128, 64)
        self.deconv1_0 = Conv1d(64, 32)
        self.warping = PointWarping()
        self.upsample = UpsampleFlow()

    @staticmethod
    def _prepare_upsample_context(xyz, sparse_xyz):
        """Compute the 3-NN interpolation map once for reused coordinates."""
        batch_size, coordinate_channels, point_count = xyz.shape
        xyz_n = xyz.permute(0, 2, 1)
        sparse_xyz_n = sparse_xyz.permute(0, 2, 1)
        knn_idx = knn_point(3, sparse_xyz_n, xyz_n)
        grouped_xyz_norm = index_points_group(sparse_xyz_n, knn_idx) - xyz_n.unsqueeze(2)
        distance = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        inverse_distance = distance.reciprocal()
        weights = inverse_distance / inverse_distance.sum(dim=2, keepdim=True)
        return (knn_idx, weights)

    @staticmethod
    def _apply_upsample_context(sparse_values, context):
        knn_idx, weights = context
        sparse_values_n = sparse_values.permute(0, 2, 1)
        grouped_values = index_points_group(sparse_values_n, knn_idx)
        dense_values = torch.sum(weights.unsqueeze(-1) * grouped_values, dim=2)
        return dense_values.permute(0, 2, 1)

    def forward(self, xyz1, xyz2, color1, color2, gt_flow, uncertainty=0.5):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1)
        color2 = color2.permute(0, 2, 1)
        pc1s, feat1s, idx1s = self.encoder(xyz1, color1)
        pc2s, feat2s, idx2s = self.encoder(xyz2, color2)
        feat1_l4_3 = self.deconv4_3(self.upsample(pc1s[3], pc1s[4], feat1s[4]))
        feat2_l4_3 = self.deconv4_3(self.upsample(pc2s[3], pc2s[4], feat2s[4]))
        if self.training:
            l2_label = index_points_gather(gt_flow, idx1s[1])
            l1_label = index_points_gather(gt_flow, idx1s[0])
            l0_label = gt_flow
        else:
            l2_label = None
            l1_label = None
            l0_label = None
        c_feat1_l3 = torch.cat([feat1s[3], feat1_l4_3], dim=1)
        c_feat2_l3 = torch.cat([feat2s[3], feat2_l4_3], dim=1)
        feat1_new_l3, feat2_new_l3, cross3 = self.cross3(pc1s[3], pc2s[3], c_feat1_l3, c_feat2_l3, feat1s[3], feat2s[3])
        feat3, flow3, certainty3 = self.flow3(pc1s[3], feat1s[3], cross3)
        if self.training:
            feat1_l3_2 = self.deconv3_2(self.upsample(pc1s[2], pc1s[3], feat1_new_l3))
        else:
            source_context_32 = self._prepare_upsample_context(pc1s[2], pc1s[3])
            feat1_l3_2 = self.deconv3_2(self._apply_upsample_context(feat1_new_l3, source_context_32))
        feat2_l3_2 = self.deconv3_2(self.upsample(pc2s[2], pc2s[3], feat2_new_l3))
        if self.training:
            up_flow2 = self.upsample(pc1s[2], pc1s[3], self.scale * flow3)
            up_certainty2 = self.upsample(pc1s[2], pc1s[3], self.scale * certainty3)
            up_feat2 = self.upsample(pc1s[2], pc1s[3], feat3)
        else:
            up_flow2 = self._apply_upsample_context(self.scale * flow3, source_context_32)
            up_certainty2 = self._apply_upsample_context(self.scale * certainty3, source_context_32)
            up_feat2 = self._apply_upsample_context(feat3, source_context_32)
        if self.training:
            flows2, feat1_new_l2, feat2_new_l2, feat2, certainty2, loss_l2 = self.recurrent2(pc1s[2], pc2s[2], feat1_l3_2, feat2_l3_2, feat1s[2], feat2s[2], up_flow2, up_feat2, l2_label, up_certainty2, uncertainty)
        else:
            flows2, feat1_new_l2, feat2_new_l2, feat2, certainty2 = self.recurrent2(pc1s[2], pc2s[2], feat1_l3_2, feat2_l3_2, feat1s[2], feat2s[2], up_flow2, up_feat2, None, up_certainty2, uncertainty)
        if self.training:
            feat1_l2_1 = self.deconv2_1(self.upsample(pc1s[1], pc1s[2], feat1_new_l2))
        else:
            source_context_21 = self._prepare_upsample_context(pc1s[1], pc1s[2])
            feat1_l2_1 = self.deconv2_1(self._apply_upsample_context(feat1_new_l2, source_context_21))
        feat2_l2_1 = self.deconv2_1(self.upsample(pc2s[1], pc2s[2], feat2_new_l2))
        if self.training:
            up_flow1 = self.upsample(pc1s[1], pc1s[2], self.scale * flows2[-1])
            up_certainty1 = self.upsample(pc1s[1], pc1s[2], self.scale * certainty2)
            up_feat1 = self.upsample(pc1s[1], pc1s[2], feat2)
        else:
            up_flow1 = self._apply_upsample_context(self.scale * flows2[-1], source_context_21)
            up_certainty1 = self._apply_upsample_context(self.scale * certainty2, source_context_21)
            up_feat1 = self._apply_upsample_context(feat2, source_context_21)
        if self.training:
            flows1, feat1_new_l1, feat2_new_l1, feat1, certainty1, loss_l1 = self.recurrent1(pc1s[1], pc2s[1], feat1_l2_1, feat2_l2_1, feat1s[1], feat2s[1], up_flow1, up_feat1, l1_label, up_certainty1, uncertainty)
        else:
            flows1, feat1_new_l1, feat2_new_l1, feat1, certainty1 = self.recurrent1(pc1s[1], pc2s[1], feat1_l2_1, feat2_l2_1, feat1s[1], feat2s[1], up_flow1, up_feat1, None, up_certainty1, uncertainty)
        if self.training:
            feat1_l1_0 = self.deconv1_0(self.upsample(pc1s[0], pc1s[1], feat1_new_l1))
        else:
            source_context_10 = self._prepare_upsample_context(pc1s[0], pc1s[1])
            feat1_l1_0 = self.deconv1_0(self._apply_upsample_context(feat1_new_l1, source_context_10))
        feat2_l1_0 = self.deconv1_0(self.upsample(pc2s[0], pc2s[1], feat2_new_l1))
        if self.training:
            up_flow0 = self.upsample(pc1s[0], pc1s[1], self.scale * flows1[-1])
            up_certainty0 = self.upsample(pc1s[0], pc1s[1], self.scale * certainty1)
            up_feat0 = self.upsample(pc1s[0], pc1s[1], feat1)
        else:
            up_flow0 = self._apply_upsample_context(self.scale * flows1[-1], source_context_10)
            up_certainty0 = self._apply_upsample_context(self.scale * certainty1, source_context_10)
            up_feat0 = self._apply_upsample_context(feat1, source_context_10)
        if self.training:
            flows0, feat1_new_l0, feat2_new_l0, feat0, certainty0, loss_l0 = self.recurrent0(pc1s[0], pc2s[0], feat1_l1_0, feat2_l1_0, feat1s[0], feat2s[0], up_flow0, up_feat0, l0_label, up_certainty0, uncertainty)
        else:
            flows0, feat1_new_l0, feat2_new_l0, feat0, certainty0 = self.recurrent0(pc1s[0], pc2s[0], feat1_l1_0, feat2_l1_0, feat1s[0], feat2s[0], up_flow0, up_feat0, None, up_certainty0, uncertainty)
        flows = [flows0[::-1], flows1[::-1], flows2[::-1], [flow3]]
        pc1 = pc1s
        pc2 = pc2s
        fps_pc1_idxs = [[None for _ in range(self.iters - 1)], [idx1s[0]], [idx1s[1]], [idx1s[2]]]
        fps_pc2_idxs = [[None for _ in range(self.iters - 1)], [idx2s[0]], [idx2s[1]], [idx2s[2]]]
        if self.training:
            return (flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2, loss_l2, loss_l1, loss_l0)
        return (flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2)


class DifFlow3DCudaGraphRunner:
    """CUDA Graph wrapper for fixed-shape no-occlusion inference.

    Inputs must be CUDA float32 tensors with shape ``[B, N, 3]``. The model
    output and full-resolution flow use graph-owned static storage and are
    overwritten by the next replay.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        batch_size: int = 1,
        num_points: int = 2048,
        uncertainty: float = 0.2,
        warmup: int = 10,
        enable_tf32: bool = True,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph inference requires CUDA.")
        if batch_size < 1 or num_points < 1 or warmup < 1:
            raise ValueError("batch_size, num_points, and warmup must be positive.")

        try:
            device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("The model has no parameters.") from error
        if device.type != "cuda":
            raise ValueError("Move the model to CUDA before creating the graph runner.")

        configure_fast_inference(enable_tf32)
        self.model = model.eval()
        self.device = device
        self.uncertainty = float(uncertainty)
        shape = (int(batch_size), int(num_points), 3)
        self.source = torch.empty(shape, device=device, dtype=torch.float32)
        self.target = torch.empty_like(self.source)
        self.gt_flow = torch.zeros_like(self.source)
        self.graph = torch.cuda.CUDAGraph()
        self.output = None
        self.predicted_flow = None
        self._capture(int(warmup))

    def _forward_static(self):
        return self.model(
            self.source,
            self.target,
            self.source,
            self.target,
            self.gt_flow,
            uncertainty=self.uncertainty,
        )

    def _capture(self, warmup: int) -> None:
        self.source.normal_(0.0, 0.25)
        self.target.normal_(0.0, 0.25)
        current = torch.cuda.current_stream(self.device)
        setup = torch.cuda.Stream(device=self.device)
        setup.wait_stream(current)
        with torch.cuda.stream(setup), torch.inference_mode():
            for _ in range(warmup):
                output = self._forward_static()
                _ = output[0][0][0].permute(0, 2, 1).contiguous()
        current.wait_stream(setup)
        torch.cuda.synchronize(self.device)

        with torch.cuda.graph(self.graph):
            with torch.inference_mode():
                self.output = self._forward_static()
                self.predicted_flow = self.output[0][0][0].permute(0, 2, 1).contiguous()
        torch.cuda.synchronize(self.device)

    def replay(self, source: torch.Tensor, target: torch.Tensor):
        """Copy a pair into static buffers, replay, and return static outputs."""
        for name, value, reference in (
            ("source", source, self.source),
            ("target", target, self.target),
        ):
            if value.device != reference.device:
                raise ValueError(f"{name} must be on {reference.device}, got {value.device}.")
            if value.dtype != reference.dtype or value.shape != reference.shape:
                raise ValueError(
                    f"{name} must have shape {tuple(reference.shape)} and dtype "
                    f"{reference.dtype}; got {tuple(value.shape)} and {value.dtype}."
                )
        self.source.copy_(source, non_blocking=True)
        self.target.copy_(target, non_blocking=True)
        self.graph.replay()
        return self.output

    def flow(self) -> torch.Tensor:
        """Return the current full-resolution flow as contiguous ``[B, N, 3]``."""
        if self.predicted_flow is None:
            raise RuntimeError("The CUDA graph has not produced an output.")
        return self.predicted_flow


def multiScaleLoss(pred_flows, gt_flow, fps_idxs, loss_f2=None, loss_f1=None, loss_f0=None, alpha=[0.02, 0.04, 0.08, 0.16]):
    num_scale = len(pred_flows)
    gt_flows = [gt_flow]
    alphas = [alpha[0]]
    a = 0
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1][0]
        if fps_idx is not None:
            sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
            gt_flows.append(sub_gt_flow)
            a += 1
            alphas.append(alpha[a])
        else:
            alphas.append(alpha[a])
    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = pred_flows[i][0].permute(0, 2, 1) - gt_flows[i]
        total_loss += alphas[i] * torch.norm(diff_flow, dim=2).sum(dim=1).mean()
        if loss_f2 is not None:
            total_loss = total_loss + alpha[2] * loss_f2.mean()
        if loss_f1 is not None:
            total_loss = total_loss + alpha[1] * loss_f1.mean()
        if loss_f0 is not None:
            total_loss = total_loss + alpha[0] * loss_f0.mean()
    return total_loss

def curvature(pc):
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim=-1, largest=False, sorted=False)
    grouped_pc = index_points_group(pc, kidx)
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim=2) / 9.0
    return pc_curvature

def computeChamfer(pc1, pc2):
    """
    pc1: B 3 N
    pc2: B 3 M
    """
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    sqrdist12 = square_distance(pc1, pc2)
    dist1, _ = torch.topk(sqrdist12, 1, dim=-1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim=1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)
    return (dist1, dist2)

def curvatureWarp(pc, warped_pc):
    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim=-1, largest=False, sorted=False)
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim=2) / 9.0
    return pc_curvature

def computeSmooth(pc1, pred_flow):
    """
    pc1: B 3 N
    pred_flow: B 3 N
    """
    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1)
    _, kidx = torch.topk(sqrdist, 32, dim=-1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx)
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim=3).sum(dim=2) / 31.0
    return diff_flow

def interpolateCurvature(pc1, pc2, pc2_curvature):
    """
    pc1: B 3 N
    pc2: B 3 M
    pc2_curvature: B 3 M
    """
    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature
    sqrdist12 = square_distance(pc1, pc2)
    dist, knn_idx = torch.topk(sqrdist12, 5, dim=-1, largest=False, sorted=False)
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx)
    norm = torch.sum(1.0 / (dist + 1e-08), dim=2, keepdim=True)
    weight = 1.0 / (dist + 1e-08) / norm
    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim=2)
    return inter_pc2_curvature

def multiScaleChamferSmoothCurvature(pc1, pc2, pred_flows, fps_idxs, iters):
    f_curvature = 0.3
    f_smoothness = 4.0
    f_chamfer = 1.0
    f_distill = 0.1
    num_scale = len(pred_flows) - iters - 1
    alpha = [0.02, 0.04, 0.08, 0.16]
    chamfer_loss = torch.zeros(1).cuda()
    smoothness_loss = torch.zeros(1).cuda()
    curvature_loss = torch.zeros(1).cuda()
    distillation_loss = torch.zeros(1).cuda()
    l = 0
    for i in range(num_scale):
        cur_flow = pred_flows[i]
        if i == 0 or (i > 0 and fps_idxs[i - 1] is not None):
            cur_pc1 = pc1[l]
            cur_pc2 = pc2[l]
            l += 1
        cur_pc2_curvature = curvature(cur_pc2)
        cur_pc1_warp = cur_pc1 + cur_flow
        dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
        moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)
        chamferLoss = dist1.sum(dim=1).mean() + dist2.sum(dim=1).mean()
        inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
        curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim=2).sum(dim=1).mean()
        chamfer_loss += alpha[l - 1] * chamferLoss
        if l < 2:
            smoothness_loss += alpha[l - 1] * computeSmooth(cur_pc1, cur_flow).sum(dim=1).mean()
        curvature_loss += alpha[l - 1] * curvatureLoss
    total_loss = f_chamfer * chamfer_loss + f_smoothness * smoothness_loss + f_curvature * curvature_loss
    return (total_loss, chamfer_loss, curvature_loss, smoothness_loss, distillation_loss)





from thop import profile, clever_format
if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    input = torch.randn((1,8192,3)).float().cuda()
    model = PointConvBidirection(iters=1).cuda()
    # #print(model)
    output = model(input,input,input,input)
    macs, params = profile(model, inputs=(input,input,input,input))
    macs, params = clever_format([macs, params], "%.3f")
    #print(macs, params)
    total = sum([param.nelement() for param in model.parameters()])
    #print("Number of parameter: %.2fM" % (total/1e6))

    from ptflops import get_model_complexity_info
    def prepare_input(resolution):
        x1 = torch.FloatTensor(1,8192,3)
        return [x1,x1,x1,x1]

    flops, params = get_model_complexity_info(model, input_res=(1, 224, 224), 
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=True)
    #print('      - Flops:  ' + flops)
    #print('      - Params: ' + params)
    # for n,p in model.named_parameters():
    #     #print(p.numel(), "\t", n, p.shape, )
    # dump_input = torch.randn((1,8192,3)).float().cuda()
    # traced_model = torch.jit.trace(model, (dump_input, dump_input, dump_input, dump_input))

    # timer = 0
    # for i in range(100):
    #     t = time.time()
    #     _ = traced_model(input,input,input,input)
    #     timer += time.time() - t
    # #print(timer / 100.0)
