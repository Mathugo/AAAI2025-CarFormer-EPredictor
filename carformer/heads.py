import torch
from torch import nn
from transformers.file_utils import ModelOutput
from transformers.activations import ACT2FN
from typing import Tuple, Dict, List, Union, Optional
from .norm import RMSNorm, LayerNormWithoutBias, LlamaRMSNorm
import math
from .utils import small_init_for_linear

class CarFormerPredictionHeadTransform(nn.Module):
    def __init__(self, config, use_rms: bool=False):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
            
        if (config.use_rmsnorm_for_heads is True) or (use_rms is True):
            self.LayerNorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.LayerNorm = LayerNormWithoutBias(config.hidden_size)
        """ SMALLINIT from Nguyen and Salazar Transformers without Tears:
Improving the Normalization of Self-Attention (2019)"""
        small_init_for_linear(self.dense, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class CarFormerEMPredictionHead(nn.Module):
    def __init__(self, config, output_dim: int=None, use_rms: bool=False):
        super().__init__()
        """
        Prediction head for Masked Event Modeling task.
        """
        self.transform = CarFormerPredictionHeadTransform(config, use_rms=use_rms)
        if output_dim is None:
            output_dim = config.vocab_size

        self.decoder = nn.Linear(config.hidden_size, output_dim, bias=True)
        """ SMALLINIT from Nguyen and Salazar Transformers without Tears:
Improving the Normalization of Self-Attention (2019)"""
        small_init_for_linear(self.decoder, config.hidden_size)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    
class CarFormerRandomPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        Prediction head to predict random events
        
        Binary classification. Random or not for each event
        """
        self.transform = CarFormerPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 2, bias=True)
        """ SMALLINIT from Nguyen and Salazar Transformers without Tears:
Improving the Normalization of Self-Attention (2019)"""
        small_init_for_linear(self.decoder, config.hidden_size)
        
    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    
class CarFormerEMRegressionHead(nn.Module):
    def __init__(self, config, use_rms: bool=False):
        super().__init__()
        """
        Prediction head for regression of Masked Event Modeling task.
        """
        self.positive_time = config.positive_time
        self.transform = CarFormerPredictionHeadTransform(config, use_rms=use_rms)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 1, bias=True)
        """ SMALLINIT from Nguyen and Salazar Transformers without Tears:
Improving the Normalization of Self-Attention (2019)"""
        small_init_for_linear(self.decoder, config.hidden_size)
        
        if config.positive_time:
            print("Positive time on regression head")
            self.softplus = nn.Softplus()

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        # no activation function, just linear output
        if self.positive_time:
            return self.softplus(hidden_states)
        return hidden_states

class CarFormerCausalPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = CarFormerEMPredictionHead(config)
        
        self.time_regression = CarFormerEMRegressionHead(config)
        self.random_prediction = CarFormerRandomPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        regression_scores = self.time_regression(sequence_output)
        random_scores = self.random_prediction(sequence_output)

        return prediction_scores, regression_scores, random_scores
    