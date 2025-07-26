# model_tg_pos.py
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from sklearn.manifold import MDS
import networkx as nx


# -------------------- 常规层 -------------------- #
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, mask=None, spatial_weights=None):
        return self.fn(self.norm(x), mask=mask, spatial_weights=spatial_weights)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mlp_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, **kwargs):  # 添加 **kwargs
        return self.net(x)



class AttentionWithSpatial(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, spatial_weights=None):
        # x: [batch*6, max_patches, dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d)->h b n d', h=self.heads), qkv)
        dots = torch.einsum('h b i d, h b j d->h b i j', q,
                            k) * self.scale  # [heads, batch*6, max_patches, max_patches]

        if spatial_weights is not None:
            # spatial_weights: [batch*6, max_patches, max_patches]
            # Ensure spatial_weights has 4 dimensions: [1, batch*6, max_patches, max_patches]
            if spatial_weights.dim() == 3:
                spatial_weights = spatial_weights.unsqueeze(0)  # [1, batch*6, 12, 12]
            elif spatial_weights.dim() == 2:
                spatial_weights = spatial_weights.unsqueeze(0).unsqueeze(-1)  # [1, batch*6, 12, 1]
                spatial_weights = spatial_weights.repeat(1, 1, 1, x.size(-1))  # [1, batch*6, 12, 12]
            else:
                raise ValueError(f"Unexpected spatial_weights dimensions: {spatial_weights.dim()}")

            # Debug: Print shapes
            print(f"dots shape: {dots.shape}, spatial_weights shape: {spatial_weights.shape}")

            if spatial_weights.shape[0] != 1:
                raise ValueError(
                    f"spatial_weights should have shape [1, batch*6, max_patches, max_patches], but got {spatial_weights.shape}")

            dots = dots + spatial_weights  # Broadcasting addition

        if mask is not None:
            # Assuming mask is broadcastable to [heads, batch*6, max_patches, max_patches]
            dots = dots.masked_fill(mask == 0, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.einsum('h b i j, h b j d->h b i d', attn, v)  # [heads, batch*6, max_patches, dim_head]
        out = rearrange(out, 'h b n d -> b n (h d)')  # [batch*6, max_patches, inner_dim]
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth=3, heads=4, mlp_mult=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, AttentionWithSpatial(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult=mlp_mult, dropout=dropout))
            ]))

    def forward(self, x, mask=None, spatial_weights=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, spatial_weights=spatial_weights) + x
            x = ff(x) + x
        return x


# -------------------- GCN -------------------- #
class GCNLayer(nn.Module):
    """
    out = ReLU(A * X * W)
    A: [batch, 6, 6], X: [batch, 6, dim]
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, A):
        # x: [batch, 6, in_dim]
        # A: [batch, 6, 6]
        out = self.linear(x)  # [batch, 6, out_dim]
        out = torch.bmm(A, out)  # [batch, 6, out_dim]
        out = F.relu(out)
        return out


class GCN(nn.Module):
    """
    可配置层数的GCN
    """

    def __init__(self, dim, layers=3):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(dim, dim) for _ in range(layers)])

    def forward(self, x, A):
        for layer in self.layers:
            x = layer(x, A)
        return x


class HTNet(nn.Module):
    def __init__(
            self,
            *,
            num_regions=6,
            max_patches_per_region=12,
            patch_dim=147,  # 7x7x3
            region_embedding_dim=32,
            intra_transformer_depth=3,
            intra_transformer_heads=4,
            inter_transformer_depth=3,
            inter_transformer_heads=4,
            gcn_layers=3,
            num_classes=3,
            dropout=0.1,
            au_adj=None
    ):
        super().__init__()
        self.num_regions = num_regions
        self.max_patches_per_region = max_patches_per_region
        self.patch_dim = patch_dim
        self.region_embedding_dim = region_embedding_dim

        # 将 AU adjacency 注册成 buffer
        # 大小为 (6,6)
        if au_adj is not None:
            self.au_adj = torch.from_numpy(au_adj).float()
        else:
            # 如果没传，就搞个单位阵做邻接
            self.au_adj = torch.eye(num_regions, dtype=torch.float32)
        self.register_buffer("AU_adj", self.au_adj)

        # patch+region 编码
        self.patch_embedding = nn.Linear(patch_dim, 256)
        self.region_embeddings = nn.Embedding(num_regions, region_embedding_dim)
        self.combine_embeddings = nn.Linear(256 + region_embedding_dim, 256)

        # 区域内Transformer
        self.intra_transformer = TransformerEncoder(
            dim=256,
            depth=intra_transformer_depth,  # 使用配置的层数
            heads=intra_transformer_heads,
            dropout=dropout
        )
        self.aggregate_intra = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
        )

        # 区域间 Transformer（如果需要）
        if inter_transformer_depth > 0:
            self.inter_transformer = TransformerEncoder(
                dim=256,
                depth=inter_transformer_depth,
                heads=inter_transformer_heads,
                dropout=dropout
            )
        else:
            self.inter_transformer = None

        # 区域间 用 GCN（只使用 AU_adj 作为邻接矩阵）
        self.gcn = GCN(dim=256, layers=gcn_layers)  # 使用配置的层数

        # 最终分类
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )

        # 可学习参数 α
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, patches, region_ids, masks=None, keypoints=None):
        """
        patches: [batch, 72, 147]
        region_ids: [batch, 72]
        masks: [batch, 72] (目前没用到，可自行做mask处理)
        keypoints: [batch, 72, 2]  # (x, y)坐标
        """
        bsz, seq_len, _ = patches.shape
        x = self.patch_embedding(patches)  # [batch,72,256]
        region_emb = self.region_embeddings(region_ids)  # [batch,72,region_embedding_dim]
        x = torch.cat([x, region_emb], dim=-1)  # [batch,72,256+region_embedding_dim]
        x = self.combine_embeddings(x)  # [batch,72,256]

        # 每个区域12个patch => reshape
        x = x.view(bsz, self.num_regions, self.max_patches_per_region, 256)  # [batch,6,12,256]
        # 同样调整 keypoints 的形状
        if keypoints is not None:
            keypoints = keypoints.view(bsz, self.num_regions, self.max_patches_per_region, 2)  # [batch,6,12,2]

        # Reshape for Transformer: [batch*6, 12, 256]
        x = x.view(bsz * self.num_regions, self.max_patches_per_region, 256)  # [batch*6, 12, 256]
        if keypoints is not None:
            keypoints = keypoints.view(bsz * self.num_regions, self.max_patches_per_region, 2)  # [batch*6, 12, 2]

        # 计算空间距离权重
        if keypoints is not None:
            # 计算欧几里得距离
            # keypoints: [batch*6, 12, 2]
            distances = torch.cdist(keypoints, keypoints, p=2)  # [batch*6, 12, 12]
            # 取倒数作为权重，添加小常数以避免除以0
            spatial_weights = 1.0 / (distances + 1e-8)  # [batch*6, 12, 12]
            # 设置无效关键点的空间权重为0
            # 无效关键点为 (0,0)，所以距离与这些关键点相关的权重也设为0
            valid_mask = (keypoints.sum(dim=-1) != 0).float()  # [batch*6, 12]
            valid_pair_mask = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(1)  # [batch*6, 12, 12]
            spatial_weights = spatial_weights * valid_pair_mask  # [batch*6, 12, 12]
        else:
            spatial_weights = None

        # 通过区域内Transformer
        x = self.intra_transformer(x, mask=None, spatial_weights=spatial_weights)  # [batch*6, 12, 256]
        x = torch.mean(x, dim=1)  # [batch*6, 256]
        x = self.aggregate_intra(x)  # [batch*6, 256]

        # reshape 回 batch维度 [batch,6,256]
        x = x.view(bsz, self.num_regions, 256)

        # 计算区域间的相似性，基于余弦相似度构建初始邻接矩阵 A
        # x: [batch, 6, 256]
        A_cos = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)  # [batch, 6, 6]

        # 将相似性转化为图结构，计算测地距离矩阵 D
        # 这里使用批处理方式，需要逐个样本计算
        D_list = []
        for i in range(bsz):
            # 构建图
            A_i = A_cos[i].detach().cpu().numpy()
            # 为避免零距离导致图不连通，设置最小权重
            A_i[A_i < 0.1] = 0.1
            G = nx.from_numpy_array(A_i)
            # 计算测地距离
            try:
                D_i = nx.floyd_warshall_numpy(G)
            except:
                # 如果图不连通，使用原始相似性矩阵的倒数作为距离
                D_i = 1 / (A_i + 1e-8)
            D_list.append(D_i)
        D = torch.tensor(np.stack(D_list), dtype=torch.float32).to(x.device)  # [batch, 6, 6]

        # 多维尺度分析（MDS）将 D 映射到低维流形空间
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
        Y_list = []
        for i in range(bsz):
            D_i = D[i].detach().cpu().numpy()
            Y_i = mds.fit_transform(D_i)
            Y_list.append(Y_i)
        Y = torch.tensor(np.stack(Y_list), dtype=torch.float32).to(x.device)  # [batch,6,2]

        # 在流形空间中计算欧几里得距离，得到最终的流形距离矩阵 M
        M = torch.cdist(Y, Y, p=2)  # [batch, 6, 6]

        # 标准化 M
        M = M / M.max(dim=-1, keepdim=True)[0].detach()

        # 将 M 与 AU共现矩阵加权相加（权重为可学习参数 α）
        # AU_adj: [6,6] -> expand to [batch,6,6]
        AU_adj = self.AU_adj.unsqueeze(0).expand(bsz, -1, -1)  # [batch,6,6]
        final_adj = self.alpha * M + (1 - self.alpha) * AU_adj  # [batch,6,6]

        # 进 GCN
        x = self.gcn(x, final_adj)  # [batch,6,256]

        # 最终平均池化 =>分类
        x = torch.mean(x, dim=1)  # [batch,256]
        logits = self.mlp_head(x)  # [batch,num_classes]
        return logits
