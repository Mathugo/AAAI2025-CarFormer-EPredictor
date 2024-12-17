from torch import nn
import torch,importlib
from transformers.activations import ACT2FN
from typing import Optional, Dict, List, Tuple
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_utils import (
    PreTrainedModel,
    PretrainedConfig)
import math
from packaging import version
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.deberta.modeling_deberta import XSoftmax
from .embedding import LlamaRotaryEmbedding, apply_rotary_pos_emb
from .utils import Cache, StaticCache, _is_package_available

@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))

@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])

@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class CarFormerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper using Rotary Positional Embedding."""

    def __init__(self, config, layer_idx: Optional[int] = None, scaling_factor: int=1):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling_factor = scaling_factor
        print("[ATTENTION2] Using scaling factor ", self.scaling_factor)
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        info_embeddings: Optional[torch.LongTensor] = None,
        query_states: Optional[torch.LongTensor] = None,
        key_states: Optional[torch.LongTensor] = None,
        value_states: Optional[torch.LongTensor] = None,
        apply_rotary: bool=True,
        output_attentions: bool = False,
        use_cache: bool = False,
        s_phase: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        """Projection matrix. Cross attention or not"""
        if query_states is not None:
            query_states = self.q_proj(query_states)
        else:
            query_states = self.q_proj(hidden_states)
            
        if key_states is not None:
            key_states = self.k_proj(key_states)
        else:
            key_states = self.k_proj(hidden_states)
        
        if value_states is not None:
            value_states = self.v_proj(value_states)
        else:
            value_states = self.v_proj(hidden_states)
            
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if info_embeddings is not None:
            info_emb = info_embeddings.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            """ Add Info Embedding """
            query_states += info_emb
            key_states += info_emb
        
        """Rotate query key"""
        if apply_rotary:
            cos, sin = self.rotary_emb(value_states, position_ids, s_phase)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.scaling_factor * self.head_dim)
        repeat_kv(key_states, self.num_key_value_groups)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        #  cast attention to correct dtype
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_probs_dropout_prob, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, None

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )