# Documentation
Examples and documentation for the `energy-guided-diffusion` repository.

## Optimizers
### Gradient Ascent
Optimize MEIs using gradient ascent.

#### Example
```python
from egg.optimizers.gradient_ascent import gradient_ascent
from egg.models import models

# Setup the parameters
seed = 0
unit = 0

# Load the model
model = models["task_driven"]["train"]

# Optimizer configuration
config = dict(
    initial={"path": "mei.initial.RandomNormal"},
    optimizer={"path": "torch.optim.AdamW", "kwargs": {"lr": 10}},
    precondition={"path": "mei.legacy.ops.GaussianBlur", "kwargs": {"sigma": 1}},
    postprocessing={"path": "mei.legacy.ops.ChangeNorm", "kwargs": {"norm": 25}},
    transparency_weight=0.0,
    stopper={"path": "mei.stoppers.NumIterations", "kwargs": {"num_iterations": 1000}},
    objectives=[
        {"path": "mei.objectives.EvaluationObjective", "kwargs": {"interval": 10}}
    ],
    device="cuda",
)

# Run the optimization
mei, score, _ = gradient_ascent(model, config, unit=unit, seed=seed)
```