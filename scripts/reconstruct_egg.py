# Imports
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torchvision.transforms import functional as TF
from tqdm import tqdm

from egg.diffusion import EGG
from egg.models import models

images = torch.Tensor(np.load("./data/75_monkey_test_imgs.npy"))
image_idxs = np.arange(0, 75)

# experiment settings
num_timesteps = 1000
energy_scale = 2  # 20
seed = 0
norm_constraint = 60  # 25
model_type = "v4_multihead_attention"  # 'task_driven' or 'v4_multihead_attention'
progressive = True


def do_run(model, energy_fn, desc="progress", grayscale=False):
    if seed is not None:
        torch.manual_seed(seed)

    cur_t = num_timesteps - 1

    samples = model.sample(energy_fn=energy_fn, energy_scale=energy_scale)

    for j, sample in enumerate(samples):
        cur_t -= 1
        if (j % 10 == 0 and progressive) or cur_t == -1:
            print()

            energy = energy_fn(sample["pred_xstart"])

            for k, image in enumerate(sample["pred_xstart"]):
                filename = f"{desc}_{0:05}.png"
                if grayscale:
                    image = image.mean(0, keepdim=True)
                image = image.add(1).div(2)

                image = image.clamp(0, 1)
                TF.to_pil_image(image).save(filename)

                tqdm.write(
                    f'step {j} | train energy: {energy["train"]:.4g} | val energy: {energy["val"]:.4g} | cross-val energy: {energy["cross-val"]:.4g}'
                )

    return energy, image


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    def energy_fn(
        x,
        target_response=None,
        val_response=None,
        cross_val_response=None,
        norm=100,
        models=None,
    ):
        tar = F.interpolate(
            x.clone(), size=(100, 100), mode="bilinear", align_corners=False
        ).mean(1, keepdim=True)

        tar = tar / torch.norm(tar) * norm  # 60
        tar = tar.clip(-1.7, 1.9)

        train_pred = models["train"](tar, data_key="all_sessions", multiplex=False)[0]
        val_pred = models["val"](tar, data_key="all_sessions", multiplex=False)[0]
        cross_val_pred = models["cross-val"](
            tar, data_key="all_sessions", multiplex=False
        )[0]

        train_energy = torch.mean((train_pred - target_response) ** 2)
        val_energy = torch.mean((val_pred - val_response) ** 2)
        cross_val_energy = torch.mean((cross_val_pred - cross_val_response) ** 2)

        return {
            "train": train_energy,
            "val": val_energy,
            "cross-val": cross_val_energy,
        }

    wandb.init(project="egg", entity="sinzlab", name=f"reconstructions_{time.time()}")
    # wandb.config.update(model_config)
    wandb.config.update(
        dict(
            energy_scale=energy_scale,
            norm_constraint=norm_constraint,
            model_type=model_type,
            image_idxs=image_idxs,
            progressive=progressive,
        )
    )

    model = EGG(num_steps=num_timesteps)

    for image_idx in image_idxs:
        print(f"Image {image_idx}")
        train_scores = []
        val_scores = []
        cross_val_scores = []
        target_image = images[image_idx].unsqueeze(0).unsqueeze(0).to(device)
        target_response = models[model_type]["train"](
            x=target_image, data_key="all_sessions", multiplex=False
        )[0]
        val_response = models[model_type]["val"](
            x=target_image, data_key="all_sessions", multiplex=False
        )[0]
        cross_val_response = models[model_type]["cross-val"](
            x=target_image, data_key="all_sessions", multiplex=False
        )[0]

        score, image = do_run(
            model=model,
            energy_fn=partial(
                energy_fn,
                target_response=target_response,
                val_response=val_response,
                cross_val_response=cross_val_response,
                norm=target_image.norm(),
                models=models[model_type],
            ),
            desc=f"progress_{image_idx}",
            grayscale=True,
        )

        wandb.log(
            {"image": wandb.Image(image), **score, "unit_idx": image_idx, "seed": seed}
        )

        train_scores.append(score["train"].item())
        val_scores.append(score["val"].item())
        cross_val_scores.append(score["cross-val"].item())

        print("Train:", train_scores)
        print("Val:", val_scores)
        print("Cross-val:", cross_val_scores)
