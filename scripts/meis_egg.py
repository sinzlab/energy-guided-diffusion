"""
Run diffusion MEIs on a set of units.
"""

import gc
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from egg.diffusion import EGG
from egg.models import models

# experiment settings
num_timesteps = 100
energy_scale = 10  # 20
seeds = [0]
norm_constraint = 25  # 25
model_type = "task_driven"  #'task_driven' #or 'v4_multihead_attention'


def do_run(model, energy_fn, desc="progress", grayscale=False, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    cur_t = num_timesteps - 1

    samples = model.sample(energy_fn=energy_fn, energy_scale=energy_scale)

    for j, sample in enumerate(samples):
        cur_t -= 1
        if (j % 10 == 0) or cur_t == -1:
            print()

            energy = energy_fn(sample["pred_xstart"])

            for k, image in enumerate(sample["pred_xstart"]):
                filename = f"{desc}_{0:05}.png"

                image = image.permute(1, 2, 0)

                # if grayscale:
                g_image = image.mean(-1, keepdim=True).clone()

                # normalize
                # tar = image / torch.norm(image) * norm_constraint * 256 / 100
                # normalize
                # tar = image / torch.norm(image) * norm_constraint * 256 / 100
                tar = image
                tar = (tar - tar.min()) / (tar.max() - tar.min())

                tqdm.write(
                    f'step {j} | train energy: {energy["train"]:.4g} | val energy: {energy["val"]:.4g} | cross-val energy: {energy["cross-val"]:.4g}'
                )

            import matplotlib.pyplot as plt

            plt.imshow(tar.cpu().detach().squeeze())#, cmap="gray", vmin=-1.7, vmax=1.7)
            plt.axis("off")
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(
                f"temp.png", transparent=True, bbox_inches="tight", pad_inches=0
            )
            plt.close()

            plt.imshow(g_image.cpu().detach().squeeze(), cmap="gray", vmin=-1.7, vmax=1.7)
            plt.axis("off")
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(
                f"temp_gray.png", transparent=True, bbox_inches="tight", pad_inches=0
            )
            plt.close()

    return energy, "temp.png", "temp_gray.png"


if __name__ == "__main__":
    data_driven_corrs = np.load("./data/data_driven_corr.npy")
    units = np.load("./data/pretrained_resnet_unit_correlations.npy")
    available_units = (data_driven_corrs > 0.5) * (units > 0.5)

    np.random.seed(42)
    units = np.random.choice(np.arange(len(available_units))[available_units], 100)
    # units = [403, 1017, 905]

    wandb.init(project="egg", entity="sinzlab", name=f"diffmeis_{time.time()}")
    wandb.config.update(
        dict(
            energy_scale=energy_scale,
            norm_constraint=norm_constraint,
            model_type=model_type,
            units=units,
        )
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else ""
                                                                     "")
    print("Using device:", device)

    def energy_fn(x, unit_idx=403, models=None):
        tar = F.interpolate(
            x.clone(), size=(100, 100), mode="bilinear", align_corners=False
        ).mean(1, keepdim=True)


        norm = tar.norm()
        # normalize
        # tar = tar / torch.norm(tar) * norm_constraint  # 60
        norm_tar = tar / torch.norm(tar) * norm_constraint  # 60

        train_energy = -models["train"](tar, data_key="all_sessions", multiplex=False)[
            0, unit_idx
        ]
        val_energy = -models["val"](norm_tar, data_key="all_sessions", multiplex=False)[
            0, unit_idx
        ]
        cross_val_energy = -models["cross-val"](
            norm_tar, data_key="all_sessions", multiplex=False
        )[0, unit_idx]

        return {
            "train": train_energy,
            "val": val_energy,
            "cross-val": cross_val_energy,
            "norm": norm,
        }

    model = EGG(num_steps=num_timesteps)

    train_scores = []
    val_scores = []
    cross_val_scores = []
    for seed in seeds:
        for unit_idx in units:
            start = time.time()
            score, image, g_image = do_run(
                model=model,
                energy_fn=partial(
                    energy_fn, unit_idx=unit_idx, models=models[model_type]
                ),
                desc=f"diffMEI_{unit_idx}",
                grayscale=False,
                seed=seed,
            )
            end = time.time()

            wandb.log(
                {
                    "image": wandb.Image(image),
                    "g_image": wandb.Image(g_image),
                    **score,
                    "unit_idx": unit_idx,
                    "seed": seed,
                    "time": end - start,
                }
            )

            train_scores.append(score["train"].item())
            val_scores.append(score["val"].item())
            cross_val_scores.append(score["cross-val"].item())

    print("Train:", train_scores)
    print("Val:", val_scores)
    print("Cross-val:", cross_val_scores)
