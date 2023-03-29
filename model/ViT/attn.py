import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AttentionLayer(nn.Module):
    def __init__(self, dim, dim_head, heads, dropout):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.attention = MultiAttention(dim)
        self.to_qury = nn.Linear(dim, inner_dim, bias=False)
        self.to_key = nn.Linear(dim, inner_dim, bias=False)
        self.to_value = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qury = self.to_qury(x).view(b, n, h, -1)
        key = self.to_key(x).view(b, n, h, -1)
        value = self.to_value(x).view(b, n, h, -1)

        out, attn = self.attention(qury, key, value)
        out = self.to_out(out)
        return out, attn


class MultiAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, q, k, v, mask=None):
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out, attn
