import torch.nn as nn
import torch
import math
from typing import *
from transformers.activations import ACT2FN
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from .norm import RMSNorm, LayerNormWithoutBias

class ContinuousEmbedding(nn.Module):
    """
    Create an embedding for continuous value by doing a linear interpolation between two discrete embedding vector
    """
    def __init__(self, max_value, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((max_value+1, embed_dim)))
        nn.init.xavier_uniform_(self.weight)
        self.max_value = max_value

    def forward(self, x: torch.FloatTensor):
        """
        x is a positive float tensor
        """
        x_int = torch.floor(x).long()
        x_frac = x - x_int
        x_frac = x_frac.unsqueeze(-1).expand_as(self.weight[x_int])
        emb_1 = self.weight[x_int]
        # Make sure we are in the good range of [0, max_value]
        emb_2 = self.weight[torch.clamp(x_int + 1, 0, self.max_value - 1)]
        # Interpolate between the two nearest embedding vectors
        emb = emb_1 * (1 - x_frac) + emb_2 * x_frac
        return emb

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, 
                 base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        self.log_10 = torch.log(torch.tensor(10))
        
    @torch.no_grad()
    def forward(self, 
                x, 
                position_ids: Optional[torch.FloatTensor] = None,
                s_phase: Optional[torch.FloatTensor] = None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        
        if s_phase is None:
            s_phase = torch.tensor(0)
            
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float() - s_phase).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)
