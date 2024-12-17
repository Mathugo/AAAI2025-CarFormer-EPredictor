from torch import nn
import torch, math
from transformers.activations import ACT2FN
from typing import Optional, Dict, List, Tuple, Union
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import (
    PreTrainedModel,
    PretrainedConfig,
    apply_chunking_to_forward,
)
from .utils import Cache, DynamicCache, StaticCache, small_init_for_linear
import torch.nn.functional as F
from .config import CarFormerConfig
from .attention import CarFormerAttention
from .norm import RMSNorm, LayerNormWithoutBias, LlamaRMSNorm

class CarFormerPreTrainedModel(PreTrainedModel):
    """
    Abstract class for CarFormer pretrained model
    """

    config_class = CarFormerConfig
    base_model_prefix = "carformer"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
                module.bias.data.add_(0.1)

    def __len__(self) -> int:
         return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CarFormerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.add_bias)
        self.dense_down = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.add_bias)

        """used the initialization scheme introduced in Wang (2021)"""
        nn.init.normal_(self.dense.weight, mean=0, std=2 / (config.max_position_embeddings * math.sqrt(config.hidden_size)))
        nn.init.normal_(self.dense_down.weight, mean=0, std=2 / (config.max_position_embeddings * math.sqrt(config.hidden_size)))

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_down(hidden_states)
        return hidden_states
    
class CarFormerLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp = CarFormerIntermediate(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CarFormerAttention(config, scaling_factor=3)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        ce_embeddings: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            ce_embeddings (`torch.FloatTensor`, *optional*):
                Context embeddings CE of size `(batch_size, sequence_length, embed_dim)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ce_embeddings=ce_embeddings,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class CarFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        """Following LLama https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        we apply rotary for each key query"""
        self.config = config
        self.layer = nn.ModuleList([CarFormerLayer(config, _) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        ce_embeddings: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        all_pos_embeddings: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_ids,
                ce_embeddings,
                output_attentions,
                all_pos_embeddings,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class CarFormerModel(CarFormerPreTrainedModel):
    def __init__(self, config: CarFormerConfig):
        super().__init__(config)
        self.config = config
        self.time_clip = config.time_clip
        self.mileage_clip = config.mileage_clip
        """Embeddings"""
        self.event_type_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings_t = nn.Linear(1, config.hidden_size)
        self.mileage_embeddings = nn.Embedding(config.mileage_km_max+1, config.hidden_size)
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.encoder = CarFormerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.event_type_embeddings

    def set_input_embeddings(self, value):
        self.event_type_embeddings = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        time: Optional[torch.tensor] = None,
        mileage: Optional[torch.tensor] = None,
        return_dict: Optional[bool] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_attention_mask: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        output_event_type_embeddings: Optional[bool] = False,
        output_context_embeddings: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        position_ids = position_ids if position_ids is not None else self.position_ids
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")

        batch_size, seq_length = input_shape
        device = input_ids.device

        """Embed"""
        inputs_embeds = self.event_type_embeddings(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        """Attention mask"""
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        """Pos Embed"""
        t_proj = self.position_embeddings_t(time.unsqueeze(-1))
        mileage_embeddings = self.mileage_embeddings(torch.clamp(mileage.long(), min=0, max=self.mileage_clip))
        ce_embeddings = t_proj + mileage_embeddings

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=causal_mask,
            head_mask=head_mask,
            position_ids=position_ids,
            ce_embeddings=ce_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.LayerNorm(sequence_output)
        
        inputs_embeds = inputs_embeds if output_event_type_embeddings is True else None
        ce_embeddings = ce_embeddings if output_context_embeddings is True else None
        causal_mask = causal_mask if output_attention_mask is True else None
        
        return (sequence_output, False, inputs_embeds, ce_embeddings, causal_mask) + encoder_outputs[1:]

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask
