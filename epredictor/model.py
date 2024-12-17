import torch, os, math, json
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, HuberLoss, L1Loss, MSELoss
from transformers.activations import ACT2FN
from ..carformer.attention import CarFormerAttention
from ..carformer.heads import CarFormerEMPredictionHead, CarFormerEMRegressionHead
from ..carformer.norm import LlamaRMSNorm, RMSNorm
from ..carformer.utils import Cache, DynamicCache, StaticCache
from .base import EPredictorForDownstream, EPredictorConfig
from typing import Callable, Dict, List, Optional, Tuple, Union

class EPredictorIntermediate(nn.Module):
    def __init__(self, config, hidden_size, bconfig):
        super().__init__()
        self.dense = nn.Linear(hidden_size, config.intermediate_size, bias=True)
        self.dense_down = nn.Linear(config.intermediate_size, hidden_size, bias=True)

        """used the initialization scheme introduced in Wang (2021)"""
        nn.init.normal_(self.dense.weight, mean=0, std=2 / (bconfig.max_position_embeddings * math.sqrt(hidden_size)))
        nn.init.normal_(self.dense_down.weight, mean=0, std=2 / (bconfig.max_position_embeddings * math.sqrt(hidden_size)))

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_down(hidden_states)
        return hidden_states

class EPredictor(EPredictorForDownstream):
    """
    Autoregressive  Error pattern predictor taking encoder (CarFormer) hiddeb representation and predicting for n tokens the next EP.

    Most of the implementation is based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    """
    def __init__(self, config: EPredictorConfig, backbone_path: str=None):
        super().__init__(config, backbone_path=backbone_path)
        self.bconfig = self.backbone.config
        self.layer_norm1 = LlamaRMSNorm(self.bconfig.hidden_size, eps=self.bconfig.rms_norm_eps)

        self.dropout1 = nn.Dropout(0.1)

        self.attention1 = CarFormerAttention(self.bconfig, scaling_factor=3)
        self.layer_norm2 = LlamaRMSNorm(self.bconfig.hidden_size, eps=self.bconfig.rms_norm_eps)
        self.intermediate1 = EPredictorIntermediate(config, self.hidden_size, self.bconfig)
        
        self.attention2 = CarFormerAttention(self.bconfig, scaling_factor=3)
        self.layer_norm3 = LlamaRMSNorm(self.bconfig.hidden_size, eps=self.bconfig.rms_norm_eps)
        self.intermediate2 = EPredictorIntermediate(config, self.hidden_size, self.bconfig)
        self.dropout2 = nn.Dropout(0.1)

        self.classifier = CarFormerEMPredictionHead(self.bconfig, output_dim=self.num_labels, use_rms=True)
        self.regressor = CarFormerEMRegressionHead(self.bconfig, use_rms=True)
        self.reg_loss = HuberLoss(reduction='mean', delta=1)
        self.ce_loss = BCEWithLogitsLoss(reduction='mean')
        self.process_time = False
        self.register_buffer("position_ids", torch.arange(self.bconfig.max_position_embeddings).expand((1, -1)))

    @torch.no_grad()
    def _build_label_time(self, time):
        return torch.log(time+1)/math.log(self.bconfig.time_h_log_base) -1
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        mileage: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        label_time: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
    ) -> Dict[str, torch.Tensor]:
        
        if self.process_time:
            time = self._build_label_time(time)
        elif self.check_process_time:
            value, ind = torch.min(time[0, :], dim=-1)
            if value.item() == 0:
                # unprocessed
                print("[EPREDICTOR] No processed input time detected, we will construct the model labeled time for regression task now ..")
                self.process_time = True
                time = self._build_label_time(time)
            self.check_process_time = False
        
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            time=time,
            mileage=mileage,
            output_context_embeddings=True,
            output_event_type_embeddings=True,
            output_attention_mask=False,
            return_dict=False
        )

        batch_size, sequence_length = input_ids.shape[:2]
        x, _, word_embeddings, ce_embeddings = outputs[:4]

        """Update attention mask"""
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + x.shape[1], device=x.device
            )
        if self.position_ids is None:
            self.position_ids = cache_position.unsqueeze(0)

        attention_mask = self._update_causal_mask(
            attention_mask, x, cache_position, past_key_values, False
        )

        hidden_states_backbone = x
        residual = word_embeddings
        
        hidden_states = self.layer_norm1(word_embeddings)

        """Attention Layer1 - self-attention layer from output event-type embedding"""
        hidden_states, self_attn_weights, present_key_value = self.attention1(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=self.position_ids,
            apply_rotary=True,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.intermediate1(hidden_states)

        residual = hidden_states
        hidden_states = self.dropout2(hidden_states)

        """Attention Layer2 - Cross attention with backbone: key, value from backbone and query from epredictor"""
        hidden_states, self_attn_weights, present_key_value = self.attention2(
            hidden_states,
            ce_embeddings=ce_embeddings,
            key_states=hidden_states_backbone,
            value_states=hidden_states_backbone,
            attention_mask=attention_mask,
            position_ids=self.position_ids,
            apply_rotary=True,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.intermediate2(hidden_states)
        hidden_states = residual + hidden_states

        """Classification Head & Regression"""
        hidden_states = self.dropout2(hidden_states)
        ep_logits = self.classifier(hidden_states)
        time_logits = self.regressor(hidden_states)

        total_loss = None
        ep_ce_loss = None
        ep_reg_loss = None
        ep_prediction = None

        ep_logits = ep_logits[:, self.min_context:, :]
        time_logits = time_logits[:, self.min_context:, :]

        if (label is not None) and (label_time is not None):
            label = label.unsqueeze(1).expand(label.size(0), self.bconfig.max_position_embeddings, label.size(-1))
            """Min Context c"""
            label = label[:, self.min_context:]
            label_time = label_time[:, self.min_context:]

            # if time is detected as raw by the backbone, we need to process it (here it's the label time)
            if self.process_time:
                label_time = self._build_label_time(label_time)

            ep_ce_loss = self.ce_loss(ep_logits.reshape(-1, self.num_labels), label.reshape(-1, self.num_labels))
            ep_reg_loss = self.reg_loss(time_logits.reshape(-1), label_time.reshape(-1))
            ep_reg_loss = torch.clamp(ep_reg_loss, max=10.0)
            total_loss = ep_ce_loss + self.alpha*ep_reg_loss
        else:
            ep_prediction = torch.sigmoid(ep_logits)
            # no need to apply an activation function for reg_loss

        return {'total_loss': total_loss, 'ce_loss': ep_ce_loss, 
                'reg_loss': ep_reg_loss, 'ep_prediction': ep_prediction, 
                'ep_logits': ep_logits, 'time_logits': time_logits,
                'time': time, 'time_label': label_time}
    
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
            self.bconfig._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask