# Imports
import os
import random
import warnings

import numpy as np
from scipy import signal


class GaussianBlur:
    """Blur an image with a Gaussian window.
    Arguments:
        sigma (float or tuple): Standard deviation in y, x used for the gaussian blurring.
        decay_factor (float): Compute sigma every iteration as `sigma + decay_factor *
            (iteration - 1)`. Ignored if None.
        truncate (float): Gaussian window is truncated after this number of standard
            deviations to each side. Size of kernel = 8 * sigma + 1
        pad_mode (string): Mode for the padding used for the blurring. Valid values are:
            'constant', 'reflect' and 'replicate'
        mei_only (True/False): for transparent mei, if True, no Gaussian blur for transparent channel:
            default should be False (also for non transparent case)
    """

    def __init__(
        self, sigma, decay_factor=None, truncate=4, pad_mode="reflect", mei_only=False
    ):
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma,) * 2
        self.decay_factor = decay_factor
        self.truncate = truncate
        self.pad_mode = pad_mode
        self.mei_only = mei_only

    def __call__(self, x, iteration=None):

        # Update sigma if needed
        if self.decay_factor is None:
            sigma = self.sigma
        else:
            sigma = tuple(s + self.decay_factor * (iteration - 1) for s in self.sigma)

        # Define 1-d kernels to use for blurring
        y_halfsize = max(int(round(sigma[0] * self.truncate)), 1)
        y_gaussian = signal.gaussian(2 * y_halfsize + 1, std=sigma[0])
        x_halfsize = max(int(round(sigma[1] * self.truncate)), 1)
        x_gaussian = signal.gaussian(2 * x_halfsize + 1, std=sigma[1])
        y_gaussian = torch.as_tensor(y_gaussian, device=x.device, dtype=x.dtype)
        x_gaussian = torch.as_tensor(x_gaussian, device=x.device, dtype=x.dtype)

        # Blur
        if self.mei_only:
            num_channels = x.shape[1] - 1
            padded_x = F.pad(
                x[:, :-1, ...],
                pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize),
                mode=self.pad_mode,
            )
        else:  # also blur transparent channel
            num_channels = x.shape[1]
            padded_x = F.pad(
                x,
                pad=(x_halfsize, x_halfsize, y_halfsize, y_halfsize),
                mode=self.pad_mode,
            )
        blurred_x = F.conv2d(
            padded_x,
            y_gaussian.repeat(num_channels, 1, 1)[..., None],
            groups=num_channels,
        )
        blurred_x = F.conv2d(
            blurred_x, x_gaussian.repeat(num_channels, 1, 1, 1), groups=num_channels
        )
        final_x = blurred_x / (y_gaussian.sum() * x_gaussian.sum())  # normalize
        # print(final_x.shape)
        if self.mei_only:
            return torch.cat(
                (final_x, x[:, -1, ...].view(x.shape[0], 1, x.shape[2], x.shape[3])),
                dim=1,
            )
        else:
            return final_x


gaussian_blur = GaussianBlur(sigma=1)

import gc
import sys
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from egg.models import models

import pickle

with open("./data/test_75_response_noisy.pkl", "rb") as f:
    data = pickle.load(f)

    images = torch.Tensor(data["images"])
    responses = torch.Tensor(data["responses"])

    # image_idxs = np.arange(0, 75)

target_l2 = torch.Tensor(np.load("./data/targets_real.npy"))
# target_l2 = torch.Tensor(np.load("./data/target_l2.npy", allow_pickle=True))
image_idxs = np.arange(0, 75)

model_type = "task_driven"  # 'task_driven' or 'v4_multihead_attention'

import time

import wandb

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
        tar = x
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

    wandb.init(
        project="egg", entity="sinzlab", name=f"gd_reconstructions_{time.time()}"
    )
    wandb.config.update(
        dict(
            model_type=model_type,
            image_idxs=image_idxs,
        )
    )

    for image_idx in image_idxs:
        print(f"Image {image_idx}")
        train_scores = []
        val_scores = []
        cross_val_scores = []

        target_response = torch.Tensor(responses[image_idx]).to(device)
        val_response = torch.Tensor(responses[image_idx]).to(device)
        cross_val_response = torch.Tensor(responses[image_idx]).to(device)
        target_image = images[image_idx].to(device)

        # optimize image to minimize energy using Adam
        class ImageGen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.image = torch.nn.Parameter(torch.randn(1, 1, 100, 100).to(device))

            def forward(self):
                return self.image

        image_gen = ImageGen().to(device)
        optimizer = torch.optim.AdamW(image_gen.parameters(), lr=0.05)

        pbar = tqdm(range(5000))
        for i in pbar:
            image = image_gen()
            image = gaussian_blur(image)
            image = image / torch.norm(image) * 60 #target_image.norm()  # 60

            res = models[model_type]["train"](
                image, data_key="all_sessions", multiplex=False
            )[0]
            val_res = models[model_type]["val"](
                image, data_key="all_sessions", multiplex=False
            )[0]
            cross_val_res = models[model_type]["cross-val"](
                image, data_key="all_sessions", multiplex=False
            )[0]

            loss = torch.mean((res - target_response) ** 2)
            val_loss = torch.mean((val_res - val_response) ** 2)
            cross_loss = torch.mean((cross_val_res - cross_val_response) ** 2)
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(
                f"loss: {loss.item()} | val_loss: {val_loss.item()} | cross_loss: {cross_loss.item()}"
            )

            if loss < target_l2[image_idx]:
                print("matched train performance")
                break

        # save image
        image = image.detach().cpu().numpy()[0, 0]
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

        wandb.log(
            {
                "image": wandb.Image(image),
                "train": loss,
                "val": val_loss,
                "cross-val": cross_loss,
                "unit_idx": image_idx,
            }
        )
