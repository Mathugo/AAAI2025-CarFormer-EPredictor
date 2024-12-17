import torch, math
from torch import nn
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN
from typing import Tuple, Dict, List, Union, Optional
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss, BCEWithLogitsLoss, HuberLoss
from .config import CarFormerConfig
from .outputs import CarFormerCausalPreTrainingOutput
from .model import CarFormerModel, CarFormerPreTrainedModel
from .heads import CarFormerCausalPreTrainingHeads

class CarFormerForPretraining(CarFormerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"time_regression.decoder.bias", 
                                        r"predictions.decoder.bias", r"time_regression.decoder.weight", 
                                        r"predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.carformer = CarFormerModel(config)
        self.head = CarFormerCausalPreTrainingHeads(config)
        self.beta = config.beta
        self.alpha = config.alpha
        self.time_h_log_base = config.time_h_log_base
        self.process_time = False
        self.check_process_time = True

        if config.reg_loss == "mse":
            self.loss_fct_reg_mean = MSELoss(reduction='mean')
            self.loss_fct_reg = MSELoss(reduction='none')
        elif config.reg_loss == "mae":
            self.loss_fct_reg_mean = L1Loss(reduction='mean')
            self.loss_fct_reg = L1Loss(reduction='none')
        elif config.reg_loss == "huber":
            self.loss_fct_reg_mean = HuberLoss(reduction='mean', delta=config.huber_loss_delta)
            self.loss_fct_reg = HuberLoss(reduction='none', delta=config.huber_loss_delta)
        else:
            raise ValueError(f'{config.reg_loss} Loss is not recognized for regression')
        self.init_weights()
    
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
                
    @torch.no_grad()
    def _build_label_time(self, time):
        return torch.log(time+1)/math.log(self.time_h_log_base) -1
        
    def next_token_prediction_forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        mileage: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        random_label: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs)-> Union[Tuple[torch.Tensor], CarFormerCausalPreTrainingOutput]:
        
        """REMOVE INPUTS IDS"""
        labels=input_ids
        random_label = torch.randint(low=0, high=2, size=(16, 258))

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.carformer(
            input_ids,
            attention_mask=attention_mask,
            mileage=mileage,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            time=time
        )
        sequence_output, _ = outputs[:2]
        lm_logits, regression_scores, random_scores = self.head(sequence_output)

        total_loss = None
        loss = None
        time_reg_loss = None
        random_loss = None
        
        """Time is raw, we have to processed it for the labeled of next time prediction"""
        # check min value
        if self.process_time:
            time = self._build_label_time(time)
        elif self.check_process_time:
            value, ind = torch.min(time[0, :], dim=-1)
            if value.item() == 0:
                # unprocessed
                print("[MODEL] No processed input time detected, we will construct the model labeled time for regression task now ..")
                self.process_time = True
                time = self._build_label_time(time)
            self.check_process_time = False

        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_time = regression_scores[..., :-1, :].contiguous()

            shift_labels = labels[..., 1:].contiguous()
            shift_labels_time = time[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            loss_fct_random = CrossEntropyLoss()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            random_loss = loss_fct_random(random_scores.view(-1, 2), random_label.view(-1))
            # remove the loss from predicting random tokens
            # Create a mask to filter out the random label
            time_reg_loss = self.loss_fct_reg(shift_time.view(-1), shift_labels_time.view(-1))

            # we shift the random label to take the correct loss for classification & reg loss
            shifted_random_label = random_label[..., 1:].contiguous()
            mask = shifted_random_label.view(-1) == 0  # Mask is True where random_label is 0
            loss = loss[mask].mean()
            time_reg_loss = time_reg_loss[mask].mean()

            total_loss = loss + self.alpha*time_reg_loss + self.beta*random_loss

        if not return_dict:
            output = (lm_logits, regression_scores, random_scores) + outputs[1:]
            return ((total_loss,) + output) if loss is not None else output

        return CarFormerCausalPreTrainingOutput(
            loss=total_loss,
            causal_loss=loss,
            reg_loss=time_reg_loss,
            random_loss=random_loss,
            prediction_logits=lm_logits,
            last_hidden_state=sequence_output,
            time_logits=regression_scores,
            random_logits=random_scores,
            labels=labels,
            time=time,
        )

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                time: Optional[torch.Tensor] = None,
                mileage: Optional[torch.Tensor] = None,
                *args, **kwargs):
        return self.next_token_prediction_forward(input_ids=input_ids, attention_mask=attention_mask, 
                                                      time=time, mileage=mileage, *args, **kwargs)
