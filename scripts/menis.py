# Imports

import gc
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import wandb
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from egg.diffusion import EGG
from egg.models import models

# experiment settings
num_timesteps = 50
energy_scale = [1, 5, 10]  # 20
seeds = [0]
norm_constraint = 50  # 50 # 25
model_type = "task_driven"  # 'task_driven' or 'v4_multihead_attention'
progressive = True

wandb.init(project="egg", entity="sinzlab", name=f"menis_{time.time()}")
# wandb.config.update(model_config)
wandb.config.update(
    dict(
        energy_scale=energy_scale,
        norm_constraint=norm_constraint,
        model_type=model_type,
        progressive=progressive,
    )
)


def do_run(
    model, energy_fn, energy_scale=1, desc="progress", grayscale=False, seed=None
):
    if seed is not None:
        torch.manual_seed(seed)

    cur_t = num_timesteps - 1

    samples = model.sample(energy_fn=energy_fn, energy_scale=energy_scale)

    for j, sample in enumerate(samples):
        cur_t -= 1
        if (j % 10 == 0 and progressive) or cur_t == -1:
            print()

            energy = energy_fn(sample["pred_xstart"])

            for k, image in enumerate(sample["sample"]):
                filename = f"sample_{desc}_{j:05}.png"
                if grayscale:
                    image = image.mean(0, keepdim=True)
                image = image.add(1).div(2)
                image = image.clamp(0, 1)

                tqdm.write(
                    f'step {j} | train energy: {energy["train"]:.4g} | val energy: {energy["val"]:.4g} | cross-val energy: {energy["cross-val"]:.4g}'
                )

    TF.to_pil_image(image).save(filename)

    return energy, image


if __name__ == "__main__":
    data_driven_corrs = np.load("./data/data_driven_corr.npy")
    units = np.load("./data/pretrained_resnet_unit_correlations.npy")
    available_units = (data_driven_corrs > 0.5) * (units > 0.5)

    np.random.seed(42)
    units = np.random.choice(np.arange(len(available_units))[available_units], 100)[:1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    def energy_fn(x, unit_idx=403, models=None):
        tar = F.interpolate(
            x.clone(), size=(100, 100), mode="bilinear", align_corners=False
        ).mean(1, keepdim=True)
        # normalize
        tar = tar / torch.norm(tar) * norm_constraint  # 60

        train_energy = -models["train"](tar, data_key="all_sessions", multiplex=False)[
            0, unit_idx
        ]
        val_energy = -models["val"](tar, data_key="all_sessions", multiplex=False)[
            0, unit_idx
        ]
        cross_val_energy = -models["cross-val"](
            tar, data_key="all_sessions", multiplex=False
        )[0, unit_idx]

        return {
            "train": train_energy,
            "val": val_energy,
            "cross-val": cross_val_energy,
        }

    model = EGG(num_steps=num_timesteps)

    for unit in units:
        for seed in seeds:
            train_scores = []
            val_scores = []
            cross_val_scores = []
            for es in energy_scale:
                score, image = do_run(
                    model=model,
                    energy_fn=partial(
                        energy_fn, unit_idx=unit, models=models[model_type]
                    ),
                    desc=f"meni_{unit}_{es}",
                    grayscale=False,
                    energy_scale=es,
                    seed=seed,
                )
                train_scores.append(score["train"].item())
                val_scores.append(score["val"].item())
                cross_val_scores.append(score["cross-val"].item())

                image = image.detach().cpu().permute(1, 2, 0).numpy()

                wandb.log(
                    {
                        "image": wandb.Image(image),
                        **score,
                        "unit_idx": unit,
                        "energy_scale": es,
                        "seed": seed,
                    }
                )

                torch.cuda.empty_cache()

    print("Train:", train_scores)
    print("Val:", val_scores)
    print("Cross-val:", cross_val_scores)
