"""
PointNet++ part segmentation model wrapper.

Based on: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
Model architecture: PointNet++ with multi-scale grouping (MSG) for part segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os


class PointNetSetAbstraction(nn.Module):
    """PointNet set abstraction layer."""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_points_concat: sample points feature data, [B, S, D']
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3] -> [B, 3, N]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, C] -> [B, C, N]

        if self.group_all:
            # For group_all, we aggregate all points into a single point
            # new_xyz should be the centroid (mean) of all points: [B, 3, 1]
            new_xyz = torch.mean(xyz, dim=2, keepdim=True)  # [B, 3, N] -> [B, 3, 1]
            
            if points is not None:
                # Concatenate along channel dimension: [B, 3, N] + [B, C, N] -> [B, 3+C, N]
                # Both tensors must have same N (number of points)
                if xyz.shape[2] != points.shape[2]:
                    # If N doesn't match, this is an error - but try to handle gracefully
                    raise ValueError(
                        f"Size mismatch in SetAbstraction: xyz has {xyz.shape[2]} points, "
                        f"but points has {points.shape[2]} points. Shapes: xyz={xyz.shape}, points={points.shape}"
                    )
                new_points = torch.cat([xyz, points], dim=1)  # [B, 3+C, N]
            else:
                new_points = xyz
            
            # For group_all, we need to reshape to 4D for Conv2d: [B, C, N] -> [B, C, 1, N]
            # This treats all N points as a single group with 1 neighbor each
            new_points = new_points.unsqueeze(2)  # [B, 3+C, N] -> [B, 3+C, 1, N]
        else:
            new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
            new_points = sample_and_group(self.radius, self.nsample, xyz, points, new_xyz)
        
        # MLP (expects 4D input [B, C, K, S] where K is number of neighbors, S is number of sampled points)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Max pooling
        if self.group_all:
            # For group_all, we have [B, C, 1, N] and want to pool over N to get [B, C, 1]
            # Pool over N dimension (dim=3) to get [B, C, 1]
            new_points = torch.max(new_points, dim=3, keepdim=False)[0]  # [B, C, 1, N] -> [B, C, 1]
            # Double-check: ensure it's exactly 3D [B, C, 1]
            while new_points.dim() > 3:
                new_points = new_points.squeeze(-1)  # Remove trailing dimensions
            if new_points.dim() == 2:
                new_points = new_points.unsqueeze(-1)  # Add dimension if 2D
        else:
            # For non-group_all, pool over K dimension (neighbors)
            new_points = torch.max(new_points, 2)[0]  # [B, C, K, S] -> [B, C, S]
        
        # For group_all, new_xyz is already [B, 3, 1], so we need to permute to [B, 1, 3]
        # For non-group_all, new_xyz needs to be permuted from [B, 3, S] to [B, S, 3]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, S] -> [B, S, 3] or [B, 3, 1] -> [B, 1, 3]
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """PointNet++ set abstraction with multi-scale grouping."""
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """PointNet feature propagation layer."""
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)  # [B, D, S] -> [B, S, D]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # points2 is [B, 1, D] after permute, we need [B, N, D]
            # Repeat along the N dimension (dim=1)
            interpolated_points = points2.repeat(1, N, 1)  # [B, 1, D] -> [B, N, D]
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


# Helper functions
def square_distance(src, dst):
    """Calculate squared euclidean distance between each two points."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """Index points by indices."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """Farthest point sampling."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B).to(device), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Query ball points."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, new_xyz=None):
    """Sample and group points."""
    B, N, C = xyz.shape
    S = npoint
    if new_xyz is None:
        new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        grouped_points_norm = torch.cat([grouped_points, grouped_xyz_norm], dim=-1)
    else:
        grouped_points_norm = grouped_xyz_norm

    grouped_points_norm = grouped_points_norm.permute(0, 3, 2, 1)
    return grouped_points_norm


class PointNet2PartSeg(nn.Module):
    """PointNet++ part segmentation model.
    
    Architecture matches the original PointNet++ part segmentation model from:
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    """
    def __init__(self, num_classes=50, normal_channel=True):
        super(PointNet2PartSeg, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        # Match original architecture exactly
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        # fp1 concatenates points1 (22 or 25 channels: cls_label[16] + l0_xyz[3] + l0_points[3 or 6])
        # with interpolated_points from l1_points (128 channels), so total is 22+128=150 or 25+128=153
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        """
        Input:
            xyz: input points position data, [B, N, 3] or [B, N, 6] (with normals)
            cls_label: class label, [B, num_classes] (one-hot encoded)
        Return:
            logits: per-point logits, [B, N, num_classes]
        """
        # Set Abstraction layers
        B, N, C = xyz.shape
        if self.normal_channel:
            l0_points_raw = xyz
            l0_xyz = xyz[:, :, 0:3]
        else:
            l0_points_raw = xyz
            l0_xyz = xyz
        l0_xyz = l0_xyz.permute(0, 2, 1)  # [B, N, 3] -> [B, 3, N]
        l0_points = l0_points_raw.permute(0, 2, 1)  # [B, N, C] -> [B, C, N]
        # Save original l0_points for fp1 (before it gets passed through sa1)
        l0_points_for_fp1 = l0_points.clone()  # [B, 3, N] or [B, 6, N]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # l1_xyz: [B, 3, 512], l1_points: [B, 320, 512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # l2_xyz: [B, 3, 128], l2_points: [B, 512, 128]
        # sa3 expects: xyz [B, N, 3], points [B, N, C] (will permute internally)
        # But sa2 returns [B, C, N], so we need to permute before passing to sa3
        l2_xyz_for_sa3 = l2_xyz.permute(0, 2, 1)  # [B, 3, 128] -> [B, 128, 3]
        l2_points_for_sa3 = l2_points.permute(0, 2, 1)  # [B, 512, 128] -> [B, 128, 512]
        # sa3 output: xyz [B, 1, 3], points [B, 1024, 1] (after group_all and permute)
        l3_xyz, l3_points = self.sa3(l2_xyz_for_sa3, l2_points_for_sa3)
        # sa3 returns [B, 1, 3] but fp3 expects [B, 3, 1], so permute back
        l3_xyz = l3_xyz.permute(0, 2, 1)  # [B, 1, 3] -> [B, 3, 1]
        # l3_points is [B, 1024, 1] but fp3 expects [B, 1024, 1] (already correct)

        # Feature Propagation layers
        # fp3 expects: xyz1 [B, 3, 128], xyz2 [B, 3, 1], points1 [B, 512, 128], points2 [B, 1024, 1]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B, 256, 128]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B, 128, 512]
        
        # Class label embedding (before fp1, as in original)
        # fp1 expects: cls_label [16] + l0_xyz [3] + l0_points [3 or 6] = 22 or 25 channels
        # Then fp1 internally concatenates with interpolated l1_points (128 channels)
        # giving total: 22+128=150 or 25+128=153 channels
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)  # [B, 16, N]
        
        # Use the saved original l0_points (not modified by sa1)
        # l0_points_for_fp1 is [B, 3, N] or [B, 6, N] (raw input, already permuted)
        fp1_input = torch.cat([cls_label_one_hot, l0_xyz, l0_points_for_fp1], dim=1)  # [B, 16+3+3/6, N] = [B, 22/25, N]
        
        # Debug: verify shapes
        expected_channels = 22 if not self.normal_channel else 25
        if fp1_input.shape[1] != expected_channels:
            raise ValueError(
                f"fp1 input has wrong number of channels: got {fp1_input.shape[1]}, "
                f"expected {expected_channels}. Shapes: cls_label={cls_label_one_hot.shape}, "
                f"l0_xyz={l0_xyz.shape}, l0_points_for_fp1={l0_points_for_fp1.shape}. "
                f"l0_points_for_fp1 should be [B, 3, N] or [B, 6, N], not expanded!"
            )
        
        l0_points = self.fp1(l0_xyz, l1_xyz, fp1_input, l1_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        logits = self.conv2(feat)
        logits = logits.permute(0, 2, 1)  # [B, N, num_classes]

        return logits


class PointNet2PartSegWrapper(nn.Module):
    """
    Wraps a pretrained PointNet++ part segmentation model for ShapeNetPart.
    
    Args:
        num_classes: number of part labels used in ShapeNetPart (typically 50).
        use_normals: whether the checkpoint expects normals (XYZ+normal = 6D input).
        checkpoint_path: path to the pretrained .pth file.
        device: 'cuda' or 'cpu'. If None, choose automatically.
    """
    def __init__(
        self,
        num_classes: int = 50,
        use_normals: bool = True,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_normals = use_normals
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model = PointNet2PartSeg(num_classes=num_classes, normal_channel=use_normals)
        self.model = self.model.to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        elif checkpoint_path:
            print(f"[PointNet2] Warning: Checkpoint not found at {checkpoint_path}")
            print("[PointNet2] Model initialized with random weights. Load checkpoint later with load_checkpoint().")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        try:
            # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
            # This is safe since we trust the checkpoint source (PointNet++ repo)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[PointNet2] ✓ Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"[PointNet2] ✗ Failed to load checkpoint: {e}")
            raise
    
    def forward(self, points, return_labels: bool = False):
        """
        Forward pass.
        
        Args:
            points: input point cloud, shape [B, N, C] where C=3 (xyz) or C=6 (xyz+normals)
            return_labels: if True, also return argmax labels
            
        Returns:
            logits: per-point logits [B, N, num_classes]
            labels: (optional) per-point labels [B, N] if return_labels=True
        """
        B, N, C = points.shape
        
        # Create dummy class label (one-hot, 16 classes for ShapeNetPart categories)
        # In practice, you might want to pass the actual category
        cls_label = torch.zeros(B, 16, dtype=torch.float32).to(self.device)
        # Set first class as default (can be overridden)
        cls_label[:, 0] = 1.0
        
        # Forward through model
        logits = self.model(points, cls_label)
        
        if return_labels:
            labels = torch.argmax(logits, dim=-1)  # [B, N]
            return logits, labels
        return logits
    
    def predict(self, points):
        """
        Convenience method: predict labels for a point cloud.
        
        Args:
            points: input point cloud, shape [B, N, C] or [N, C] (will add batch dim)
            
        Returns:
            labels: per-point labels [B, N] or [N]
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)  # Add batch dimension
        
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        
        points = points.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            logits, labels = self.forward(points, return_labels=True)
        
        if labels.shape[0] == 1:
            labels = labels.squeeze(0)  # Remove batch dimension if single sample
        
        return labels.cpu().numpy()


def load_pretrained_model(
    checkpoint_path: str,
    num_classes: int = 50,
    use_normals: bool = True,
    device: Optional[str] = None,
) -> PointNet2PartSegWrapper:
    """
    Construct the model and load pretrained weights from checkpoint_path.
    
    Args:
        checkpoint_path: path to the pretrained .pth checkpoint file
        num_classes: number of part classes (default 50 for ShapeNetPart)
        use_normals: whether model expects normals (6D input)
        device: 'cuda' or 'cpu', or None for auto-detect
        
    Returns:
        PointNet2PartSegWrapper instance with loaded weights
    """
    model = PointNet2PartSegWrapper(
        num_classes=num_classes,
        use_normals=use_normals,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return model

