from transformers.file_utils import ModelOutput
from transformers.modeling_utils import (
    PreTrainedModel,
    PretrainedConfig,
    apply_chunking_to_forward,
)
from transformers.activations import ACT2FN
from .norm import RMSNorm

class CarFormerConfig(PretrainedConfig):
    """
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    Example:

    ```python
    >>> from config import CarFormerConfig
    >>> from pretraining import CarFormerForPretraining

    >>> # Initializing a CarFormer style configuration
    >>> configuration = CarFormerConfig()

    >>> model = CarFormerForPretraining(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
     ```
    """
    model_type = "carformer"
    
    def __init__(
        self,
        vocab_size=20000,
        hidden_size=600,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=600*4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_bias=True,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=258,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        huber_loss_delta=1,
        use_return_dict=False,
        use_cache=False,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        add_bias=True,
        positive_time=False,
        
        num_key_value_heads=12,
        pretraining_tp=1,
        mileage_log_base=8,
        time_h_log_base=10,
        time_clip=30*24,
        mileage_km_max=300,
        mileage_clip=300,
        reg_loss="huber",
        alpha=1,
        beta=1,
        classifier_dropout=None,
        use_rmsnorm_for_heads=True,
        pad_token_id=0,
        **kwargs,
         ):
        super().__init__(
                pad_token_id=pad_token_id,
                **kwargs,
        )
        self.add_bias = add_bias
        self.alpha = alpha
        self.beta = beta
        self.huber_loss_delta = huber_loss_delta
        self.time_clip = time_clip
        self.mileage_km_max = mileage_km_max
        self.mileage_log_base = mileage_log_base
        self.mileage_clip = mileage_clip
        self.time_h_log_base = time_h_log_base
        self.reg_loss = reg_loss
        # Output positive time for time masking
        self.positive_time = positive_time
        self.use_rmsnorm_for_heads = use_rmsnorm_for_heads
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_key_value_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.pretraining_tp = pretraining_tp
        
    def check_config(self):
        assert not (self.hidden_size % self.num_attention_heads), "Hidden size must be / by the number of head"
