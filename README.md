# 🥚 EGG: Energy Guided Diffusion for optimizing neurally exciting images 
Diffusion Induced Most Exciting inputs

<img src="./assets/menis.png">

*This repository is based on the [guided-diffusion](https://github.com/openai/guided-diffusion) repository.*

# Installation
## Package Requirements
You can install the required packages by running:
```bash
pip install -e .
```

To run neural experiments you need to install the [nnvision](https://github.com/sinzlab/nnvision.git) package.
```bash
mkdir lib
git clone -b model_builder https://github.com/sinzlab/nnvision.git ./lib/nnvision
pip install -e ./lib/nnvision
```

## Pre-trained model weights
To run EGG you need to download the pre-trained weights of the ADM model.
The experiments use a model pretrained by OpenAI on 256x256 ImageNet images.

| Model                   | Weights |
|-------------------------| --- |
| ImageNet 256x256 uncond | [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) |

place the weights in the `models` folder.

# Usage
Here is a minimal example for running the EGG diffusion on a pretrained model.

```python
from functools import partial

from egg.models import models
from egg.diffusion import EGG

# Setup the parameters
energy_scale = 5
num_samples = 1
num_steps = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def energy_fn(pred_x_0, unit_idx=0):
    """
    Energy function for optimizing MEIs, i.e. images that maximally excite a given unit.

    :param pred_x_0: the predicted "clean" image
    :param unit_idx: the index of the unit to optimize
    :return: the neural of the predicted image for the given unit
    """
    return dict(train=models['task_driven']['train'](pred_x_0)[..., unit_idx])


diffusion = EGG(
    diffusion_artefact='./models/256x256_diffusion_uncond.pt',
    num_steps=num_steps
)
samples = diffusion.sample(
    energy_fn=partial(energy_fn, unit_idx=0),
    energy_scale=energy_scale,
    num_samples=1,
    device=device,
)
```