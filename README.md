# AAAI2025-CarFormer-EPredictor

>📋 Model code & Evaluation metrics for Vehicle Error Patterns Predictive Maintenance using autoregressive Transformer models

# Harnessing Event Sensory Data for Error Pattern Prediction in Vehicles: A Language Model Approach

This repository is the official implementation of [Harnessing Event Sensory Data for Error Pattern Prediction in Vehicles: A Language Model Approach](no_preprint_from_now). 

>📋 [] PLOTS Car.pdf
>
## Requirements

To install requirements:

```setup
pip install transformers pytorch packaging einops dataclasses json 
```

## Instantiate

To instantiate the model(s) in the paper, run this command:

```
from paper_code.carformer.pretraining import CarFormerForPretraining
from paper_code.carformer.config import CarFormerConfig
from paper_code.epredictor.model import EPredictor
from paper_code.epredictor import model
from paper_code.epredictor.base import EPredictorConfig
```

### CarFormer
```
config = CarFormerConfig()
model = CarFormerForPretraining(config)
model
model.save_pretrained('carformer_test')
```

### EPredictor
```
config = EPredictorConfig('carformer_test',
                         alpha=1,
                         min_context=30,
                         label_mapping={'ep1': 0},
                         )
model = EPredictor(config)
model.save_pretrained('epredictor_test', None)
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

CPMW: The Confident Predictive Maintenance Window Area Under The Curve was used to access which model configuration was the best for predictive maintenance of EPs happenning to the vehicle. 

```eval
def calc_CPMWAUC(y, seq_len_mean: int, 
                 min_context: int,
                 confidence_threshold: float, 
                 mode: str='decrease'):
    """
    Calculate the area under the curve for some scores in list `t` for values >= `theta`
    and <= some score at index `x`.

    Parameters:
    - y (list of float): List of metric values.
    - confidence_threshold (float): Threshold value to enter the CPMW.
    - seq_len_mean (int): Index in list `y` up to which the metric is considered into the calculation.
    - min_context (int): Context c of the EPredictor which generated the metric values.
    - mode (str): Weither we calculate the CPMWAUC for values that decrease or increase. Typically, for MAE like metrics, we choose decrease, whereas for classification metrics we should increase.

    Returns:
    - float: Area under the curve.
    """

    print(len(y))
    y = min_context * [0.0] + y 
    print(len(y))

    # Validate input
    if not 0 <= seq_len_mean < len(y):
        raise ValueError("Index x is out of bounds.")

    # Get the score at index x
    zeta_x = y[seq_len_mean]
    print("Metric of x mean", zeta_x)

    # Filter the scores based on the constraints
    if mode == 'increase':
        filtered_y = [(i, score) for i, score in enumerate(y) if confidence_threshold <= score <= zeta_x]
    elif mode == 'decrease':
        # for regression 
        filtered_y = [(i, score) for i, score in enumerate(y) if confidence_threshold >= score >= zeta_x]

    if not filtered_y:
        return 0.0

    # Separate the indices and the scores
    indices, scores = zip(*filtered_y)

    # Calculate the area using the trapezoidal rule
    area = np.trapz(scores, indices)
    return area
```

## Contribution
>📋  Apache License 2.0
