from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch

@dataclass
class CarFormerCausalPreTrainingOutput(ModelOutput):
    """
    Output type of [`ModelOutput`].
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    time_logits: Optional[Tuple[torch.FloatTensor]] = None
    random_logits: Optional[Tuple[torch.FloatTensor]] = None
    time: Optional[Tuple[torch.FloatTensor]] = None
    labels: Optional[Tuple[torch.LongTensor]] = None
    causal_loss: Optional[Tuple[torch.LongTensor]] = None
    reg_loss: Optional[Tuple[torch.LongTensor]] = None
    random_loss: Optional[Tuple[torch.LongTensor]] = None
    last_hidden_state: Optional[Tuple[torch.LongTensor]] = None