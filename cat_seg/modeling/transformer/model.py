# --------------------------------------------------------
# CAT-Seg: Cost Aggregation for Open-vocabulary Semantic Segmentation
# Licensed under The MIT License [see LICENSE for details]
# Written by Seokju Cho and Heeseong Shin
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from cca_zoo.models import CCA
from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert
from PIL import Image
import cv2
import os
# Modified Swin Transformer blocks for guidance implementetion
# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, appearance_guidance_dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.k = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, attn_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        q = self.q(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x[:, :, :self.dim]).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, dim, appearance_guidance_dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, appearance_guidance_dim=appearance_guidance_dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, appearance_guidance):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if appearance_guidance is not None:
            appearance_guidance = appearance_guidance.view(B, H, W, -1)
            x = torch.cat([x, appearance_guidance], dim=-1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, x_windows.shape[-1])  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


#####  这个操作对cost volume没有任何影响， 通道数，尺寸大小都是一样的    #####
class SwinTransformerBlockWrapper(nn.Module):
    def __init__(self, dim, appearance_guidance_dim, input_resolution, nheads=4, window_size=5, pad_len=0):
        super().__init__()
        self.block_1 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=0)
        self.block_2 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=window_size // 2)
        self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, appearance_guidance_dim)) if pad_len > 0 and appearance_guidance_dim > 0 else None
    
    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
            appearance_guidance: B C H W
        """
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'B C T H W -> (B T) (H W) C')
        if appearance_guidance is not None:
            appearance_guidance = self.guidance_norm(repeat(appearance_guidance, 'B C H W -> (B T) (H W) C', T=T))
        x = self.block_1(x, appearance_guidance)
        x = self.block_2(x, appearance_guidance)
        x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, T=T, H=H, W=W)
        return x


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type='linear'):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        if attention_type == 'linear':
            self.attention = LinearAttention()
        elif attention_type == 'full':
            self.attention = FullAttention()
        else:
            raise NotImplementedError
    
    def forward(self, x, guidance):
        """
        Arguments:
            x: B, L, C
            guidance: B, L, C
        """
        q = self.q(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.q(x)
        k = self.k(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.k(x)
        v = self.v(x)

        q = rearrange(q, 'B L (H D) -> B L H D', H=self.nheads)
        k = rearrange(k, 'B S (H D) -> B S H D', H=self.nheads)
        v = rearrange(v, 'B S (H D) -> B S H D', H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, 'B L H D -> B L (H D)')
        return out


########   尺寸不变  #######
class ClassTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=64, guidance_dim=64, nheads=8, attention_type='linear', pooling_size=(4, 4), pad_len=256) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pooling_size) if pooling_size is not None else nn.Identity()
        self.attention = AttentionLayer(hidden_dim, guidance_dim, nheads=nheads, attention_type=attention_type)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, guidance_dim)) if pad_len > 0 and guidance_dim > 0 else None
    
    def pool_features(self, x):
        """
        Intermediate pooling layer for computational efficiency.
        Arguments:
            x: B, C, T, H, W
        """
        B = x.size(0)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.pool(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x

    def forward(self, x, guidance):
        """
        Arguments:
            x: B, C, T, H, W
            guidance: B, T, C
        """
        B, C, T, H, W = x.size()
        x_pool = self.pool_features(x)
        *_, H_pool, W_pool = x_pool.size()
        
        if self.padding_tokens is not None:
            orig_len = x.size(2)
            if orig_len < self.pad_len:
                # pad to pad_len
                padding_tokens = repeat(self.padding_tokens, '1 1 C -> B C T H W', B=B, T=self.pad_len - orig_len, H=H_pool, W=W_pool)
                x_pool = torch.cat([x_pool, padding_tokens], dim=2)

        x_pool = rearrange(x_pool, 'B C T H W -> (B H W) T C')
        if guidance is not None:
            if self.padding_guidance is not None:
                if orig_len < self.pad_len:
                    padding_guidance = repeat(self.padding_guidance, '1 1 C -> B T C', B=B, T=self.pad_len - orig_len)
                    guidance = torch.cat([guidance, padding_guidance], dim=1)
            guidance = repeat(guidance, 'B T C -> (B H W) T C', H=H_pool, W=W_pool)

        x_pool = x_pool + self.attention(self.norm1(x_pool), guidance) # Attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool)) # MLP

        x_pool = rearrange(x_pool, '(B H W) T C -> (B T) C H W', H=H_pool, W=W_pool)
        x_pool = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=True)
        x_pool = rearrange(x_pool, '(B T) C H W -> B C T H W', B=B)

        if self.padding_tokens is not None:
            if orig_len < self.pad_len:
                x_pool = x_pool[:, :, :orig_len]

        x = x + x_pool # Residual
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def cca(feature_list):
    flat_features = []
    for f in feature_list:
        B, C, H, W = f.shape
        f_flat = f.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (N, D)
        flat_features.append(f_flat.cpu().numpy())

    # === 计算两两之间的 CCA 相似度 ===
    num_experts = len(flat_features)
    cca_matrix = np.zeros((num_experts, num_experts))

    for i in range(num_experts):
        for j in range(num_experts):
            if i == j:
                cca_matrix[i, j] = 1.0  # 自身相似度为1
            else:
                X = flat_features[i]
                Y = flat_features[j]

                # CCA 计算
                model = CCA(latent_dims=min(X.shape[1], Y.shape[1], 128))  # 取前10个典型维度
                model.fit((X, Y))
                corr = model.score((X, Y))  # 返回每个典型相关系数

                cca_matrix[i, j] = np.mean(corr)  # 平均作为相似度

    print("CCA 相似度矩阵：")
    print(cca_matrix)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ChannelAttentionFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # 输入两个特征图拼接后为 2C 通道，做通道注意力生成每个通道的融合权重
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),              # B × 2C × 1 × 1 × 1
            nn.Conv3d(2 * channel, channel, 1),   # B × C × 1 × 1 × 1
            nn.Sigmoid()
        )

    def forward(self, feat_spatial, feat_class):
        # feat_spatial 和 feat_class: B × C × T × H × W
        fused = torch.cat([feat_spatial, feat_class], dim=1)  # B × 2C × T × H × W
        attn = self.attn(fused)  # B × C × 1 × 1 × 1

        out = attn * feat_spatial + (1 - attn) * feat_class  # B × C × T × H × W
        return out






class DualFeatureMoE(nn.Module):
    def __init__(self, dim, experts=4, reduction_ratio=4):
        super().__init__()
        self.experts = experts
        
        # 特征投影增强差异性
        self.spatial_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.class_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # 多专家结构增强  每个专家从特定的角度来学习特定的知识
        self.expert_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim*2, dim*2, kernel_size=1),
                nn.SyncBatchNorm(dim*2),
                nn.GELU(),
                nn.Conv2d(dim*2, dim, kernel_size=3, padding=1, groups=dim),
                nn.GELU()
            ) for _ in range(experts)
        ])

        # 空间感知门控机制 主要是学习每个专家的权重
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim*2, dim // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // reduction_ratio, experts, kernel_size=1),
            nn.Softmax(dim=1)  # Softmax across expert dimension
        )

        # 残差连接
        self.residual = nn.Conv2d(dim*2, dim, kernel_size=1)
        self.res_scale = nn.Parameter(torch.tensor(0.5))  # 可学习残差比例

    def forward(self, spatial_feat, class_feat, batched_inputs=None):
        B, C, T, H, W = spatial_feat.shape

        # reshape to (B*T, C, H, W)
        x_spatial = spatial_feat.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        x_class   = class_feat.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

        # 特征投影
        x_spatial = self.spatial_proj(x_spatial)
        x_class = self.class_proj(x_class)

        # concat: (B*T, 2C, H, W)
        expert_input = torch.cat([x_spatial, x_class], dim=1)

        # 获取空间门控权重: (B*T, experts, H, W)
        gate_weights = self.gate_net(expert_input)
        
        # 多专家融合
        fused = 0
        feature_list = []
        for i, expert in enumerate(self.expert_layers):
            expert_out = expert(expert_input)  # (B*T, C, H, W)
            feature_list.append(expert_out)
            weight = gate_weights[:, i:i+1, :, :]  # (B*T, 1, H, W)
            fused += expert_out * weight
        # cca(feature_list)
        # 残差连接
        residual = self.residual(expert_input)  # (B*T, C, H, W)
        fused = self.res_scale * residual + fused

        # reshape 回原始格式: (B, C, T, H, W)
        fused = fused.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return fused


def cosine_disentanglement_loss(F_c, F_s, eps=1e-6):
    """
    Input:
        F_c, F_s: (B, C, T, H, W) feature tensors
    Output:
        loss: scalar tensor
    """
    B, C, T, H, W = F_c.shape

    # reshape to (N, C) where N = B * T * H * W
    Fc = F_c.permute(0, 2, 3, 4, 1).reshape(-1, C)
    Fs = F_s.permute(0, 2, 3, 4, 1).reshape(-1, C)

    # Normalize
    Fc = Fc / (Fc.norm(dim=1, keepdim=True) + eps)
    Fs = Fs / (Fs.norm(dim=1, keepdim=True) + eps)

    # Cosine similarity
    cos_sim = (Fc * Fs).sum(dim=1)  # shape: (N,)

    # Loss: square to encourage orthogonality (cos ≈ 0)
    loss = (cos_sim ** 2).mean()

    return loss , cos_sim.mean()



class AggregatorLayer(nn.Module):  # 3.1 M
    def __init__(self, hidden_dim=64, text_guidance_dim=512, appearance_guidance=512, nheads=4, input_resolution=(20, 20), pooling_size=(5, 5), window_size=(10, 10), attention_type='linear', pad_len=256) -> None:
        super().__init__()
        self.swin_block_first = SwinTransformerBlockWrapper(hidden_dim, appearance_guidance, input_resolution, nheads, window_size)
        self.swin_block_second = SwinTransformerBlockWrapper(hidden_dim, appearance_guidance, input_resolution, nheads, window_size)
        self.fusion_by_channel_first = DualFeatureMoE(hidden_dim)
        self.attention_first = ClassTransformerLayer(hidden_dim, text_guidance_dim, nheads=nheads, attention_type=attention_type, pooling_size=pooling_size, pad_len=pad_len)
        self.attention_second = ClassTransformerLayer(hidden_dim, text_guidance_dim, nheads=nheads, attention_type=attention_type, pooling_size=pooling_size, pad_len=pad_len)
        self.fusion_by_channel_second = DualFeatureMoE(hidden_dim)
    

    def forward(self, x, appearance_guidance, text_guidance,batched_inputs,need_loss=False):
        """
        Arguments:
            x: B C T H W
        """
        x_spatial_first = self.swin_block_first(x, appearance_guidance)  # 产生第一个空间聚合  B C T H W
        x_class_first = self.attention_first(x, text_guidance)  # 产生第一个类聚合  B C T H W 
        feature_by_channel = self.fusion_by_channel_first(x_spatial_first,x_class_first)
        x_spatial_second = self.swin_block_second(feature_by_channel,appearance_guidance)
        x_class_second = self.attention_second(feature_by_channel,text_guidance)
        feature_by_channel_second = self.fusion_by_channel_second(x_spatial_second,x_class_second,batched_inputs)

        if need_loss:
            loss_Orthogonality_1, cos_sim_1 = cosine_disentanglement_loss(x_spatial_first, x_class_first)
            loss_Orthogonality_2, cos_sim_2 = cosine_disentanglement_loss(x_spatial_second, x_class_second)

            return feature_by_channel_second, (loss_Orthogonality_1 + loss_Orthogonality_2) / 2.0, (cos_sim_1 + cos_sim_2) / 2.0
        else:
            return feature_by_channel_second










class AggregatorResNetLayer(nn.Module):
    def __init__(self, hidden_dim=64, appearance_guidance=512) -> None:
        super().__init__()
        self.conv_linear = nn.Conv2d(hidden_dim + appearance_guidance, hidden_dim, kernel_size=1, stride=1)
        self.conv_layer = Bottleneck(hidden_dim, hidden_dim // 4)


    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
        """
        B, T = x.size(0), x.size(2)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        appearance_guidance = repeat(appearance_guidance, 'B C H W -> (B T) C H W', T=T)

        x = self.conv_linear(torch.cat([x, appearance_guidance], dim=1))
        x = self.conv_layer(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, guidance=None):
        x = self.up(x)
        if guidance is not None:
            T = x.size(0) // guidance.size(0)
            guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, guidance], dim=1)
        return self.conv(x)


class Aggregator(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers # 2
        self.hidden_dim = hidden_dim # 128

        ### hidden_dim = 128   text_guidance_proj_dim = 128  appearance_guidance_proj_dim = 128   nheads = 4
        ### feature_resolution = [24 24]  pooling_size = [2,2]  window_size = 12    attention_type = linear    pad_len = 256
        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ]) 



        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance,batched_inputs,need_loss=False):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C)
            apperance_guidance: tuple of (B, C, H, W)
        """
        classes = None

        # total_params = sum(p.numel() for p in self.layers.parameters() if p.requires_grad)
        # print(f"Total Trainable Parameters: {total_params / 1e6:.2f} M")

        corr = self.correlation(img_feats, text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            text_feats = th_text
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)
        #corr = self.feature_map(img_feats, text_feats)
        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        # for layer in self.layers:
        #     corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        loss = 0
        smi = 0 
        if need_loss:
            for layer in self.layers:
                corr_embed,loss_1, Similarity = layer(corr_embed, projected_guidance, projected_text_guidance,need_loss)
                loss = loss + loss_1
                smi = smi + Similarity
            loss = 0.001 *  loss / self.num_layers
        else:
            for layer in self.layers:
                corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance,batched_inputs)


        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)  # B C H W
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        # return logit
        if need_loss:
            return logit , loss, smi / 2
        else:
            return logit

