from ..carformer.model import CarFormerModel
from transformers import PreTrainedModel, PretrainedConfig, AutoModel
from typing import Dict, Tuple, List, Union, Callable, Optional
import torch, os, json
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss, HuberLoss
class EPredictorConfig(PretrainedConfig):
    """
    Configuration class for the EPredictor model.

    Arguments:
        local_backbone_path (`str`, optional):
            Path to the CarFormer backbone to use while fine-tuning. Default is `None`.
        
        alpha (`float`, optional):
            Coefficient to balance the loss between classification loss and regression loss. Default is `1`.
        
        freeze_backbone (`bool`, optional): 
            Whether to freeze the backbone layer during training. Default is `True`.
        
        intermediate_size (`int`, optional): 
            Intermediate size of the MLP (Multi-Layer Perceptron) layer. Default is `1200`.
        
        initializer_range (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices. Default is `0.02`.
        
        label_mapping (`Dict[str, int]`, optional):
            A dictionary mapping labels to integers for error patterns. Default is `None`.
        
        min_context (`int`, optional):
            Minimum context size c for the model. Default is `30`.
        
        num_heads (`int`, optional):
            Number of attention heads in the model. Default is `4`.
        
        add_bias (`bool`, optional):
            Whether to add a bias term to layers. Default is `True`.
        
        hidden_act (`str`, optional):
            The activation function to use in the hidden layers. Default is `'gelu'`.
                
        normalization (`str`, optional):
            The type of normalization to use. Default is `'layernorm'`.
        
        **kwargs:
            Additional keyword arguments passed to `PretrainedConfig`.
    """
    def __init__(self,
                 local_backbone_path: str = None,
                 alpha: float = 1,
                 freeze_backbone: bool = True,
                 intermediate_size: int = 1200,
                 initializer_range: float = 0.02,
                 label_mapping: Dict[str, int] = None,
                 min_context: int = 30,
                 num_heads: int = 4,
                 add_bias: bool = True,
                 hidden_act: str = 'gelu',
                 normalization: str = "layernorm",
                 **kwargs):
        super().__init__(**kwargs)
        self.local_backbone_path = local_backbone_path
        self.alpha = alpha
        self.add_bias = add_bias
        self.normalization = normalization
        self.freeze_backbone = freeze_backbone
        self.initializer_range = initializer_range
        self.label_mapping = label_mapping
        self.min_context = min_context
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        
    def check_config(self):
        pass

class EPredictorForDownstream(PreTrainedModel):
    """
    Abstract class to handle weights initialization and base class for downstream task CCM Predictor
    """
    config_class = EPredictorConfig
    base_model_prefix = "eprediction"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = False

    def __init__(self, config, backbone_path: str=None):
        super().__init__(config)
        
        if backbone_path is not None:
            local_path = backbone_path
        else:
            local_path = config.local_backbone_path
        print(f'Loading backbone {local_path} ..')
        self.backbone = CarFormerModel.from_pretrained(local_path).to('cuda')
        self.bconfig = self.backbone.config
        self.config = config
        self.num_labels = len(config.label_mapping)
        # update for when we will save it
        self.config.backbone_config = self.bconfig
        # Get the number of hidden units from the backbone model
        self.hidden_size = self.bconfig.hidden_size
        print(f"Hidden size of backbone: {self.hidden_size}")
        print(f"Number of labels for classifier {self.num_labels}")
        print(f"Min context {config.min_context}")
        self.min_context = config.min_context
        self.reg_loss = HuberLoss(reduction='mean')
        self.alpha = config.alpha
        self.max_len = self.bconfig.max_position_embeddings
        self.check_process_time = True
        self.register_buffer("bias", torch.tril(torch.ones(self.max_len, self.max_len) * -1e9), persistent=False)
        if config.freeze_backbone:
            print(f"Freezing backbone ..")
            for param in self.backbone.parameters():
                param.requires_grad = False
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

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        data_params: Dict[str, str],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        
        if hasattr(self.config, 'backbone_config'):
            print("Converting backbone config into dict ..")
            if type(self.config.backbone_config) is not dict:
                try:
                    self.config.backbone_config = self.config.backbone_config.to_dict()
                except:
                    pass
        
        print("Saving epredictor to ", save_directory)
        super().save_pretrained(save_directory,
                                is_main_process=is_main_process,
                                state_dict=state_dict,
                                push_to_hub = push_to_hub,
                                max_shard_size = max_shard_size,
                                safe_serialization= safe_serialization,
                                variant = variant,
                                token = token,
                                save_peft_format = save_peft_format,
                                **kwargs,)
        print("Saving backbone ..")
        self.backbone.save_pretrained(os.path.join(save_directory, 'backbone'))
        
        print("Saving data parameters used ..")
        if data_params is not None:
            if 'tokenizer' in data_params:
                # non serializable
                data_params.pop('tokenizer')
                with open(f'{save_directory}/data_config.json', 'w') as json_file:
                   # Write the dictionary to the file in JSON format
                   json.dump(data_params, json_file)
    
    def _expand_attention_mask_vanilla(self, attention_mask, num_heads):
        # Example binary attention_mask [batch_size, seq_len]
        batch_size, seq_len = attention_mask.size()

        # Create an expanded mask [batch_size, seq_len, seq_len]
        # Initialize with zeros (indicating full attention allowance)
        expanded_attention_mask = torch.zeros(
            (batch_size * num_heads,
             seq_len, seq_len),
            device=attention_mask.device)

        #Fill in the expanded mask
        for i in range(batch_size):
            seq_mask = attention_mask[i].unsqueeze(0).repeat(num_heads, 1, 1)  # [num_heads, seq_len, seq_len]
            expanded_attention_mask[i * num_heads:(i + 1) * num_heads] = seq_mask * seq_mask.transpose(1, 2)

        # Wherever the attention_mask was 0 (padding), set to -1e4 to block attention
        expanded_attention_mask = expanded_attention_mask.masked_fill(expanded_attention_mask == 0, float('-1e4'))
        return expanded_attention_mask

    def __len__(self) -> int:
         print(f"[MODEL] {sum(p.numel() for p in self.parameters() if p.requires_grad)} parameters requires gradients")