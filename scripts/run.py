# Imports

import gc
import sys

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter
from lib.nnvision.nnvision.models.trained_models.v4_task_driven import task_driven_ensemble_1

from PIL import Image

from lib.nnvision.nnvision.models.trained_models.v4_data_driven import \
    v4_multihead_attention_ensemble_model


sys.path.append('./guided-diffusion')

target = np.load('cute_monkey_60k_responses.npy')
target = torch.from_numpy(target).float().cuda()

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

# Model settings
model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '100',  # Modify this value to decrease the number of timesteps.
    'image_size': 256,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_checkpoint': False,
    'use_fp16': True,
    'use_scale_shift_norm': True,
})

batch_size = 1
tv_scale = 150  # Controls the smoothness of the final output.
n_batches = 1
init_image = None  # This can be an URL or Colab local path and must be in quotes.
skip_timesteps = 0  # This needs to be between approx. 200 and 500 when using an init image.
# Higher values make the output look more like the init.
seed = 0


def do_run(model, diffusion, target_image):
    if seed is not None:
        torch.manual_seed(seed)

    def energy_fn(x):
        tar = F.interpolate(x.clone(), size=(100, 100), mode='bilinear', align_corners=False).mean(1, keepdim=True)

        # normalize
        tar = tar / torch.norm(tar) * 100  # 60

        response = task_driven_ensemble_1(tar, data_key="all_sessions", multiplex=True)[0]

        return torch.mean((response[::49] - target[0, ::49]) ** 2)

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    for i in range(n_batches):
        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        samples = sample_fn(
            model,
            (batch_size, 3, model_config['image_size'], model_config['image_size']),
            clip_denoised=False,
            model_kwargs={},
            progress=True,
            cond=target_image,
            energy_fn=energy_fn,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % 10 == 0 or cur_t == -1:
                print()
                for k, image in enumerate(sample['pred_xstart']):
                    filename = f'progress_{0:05}.png'
                    # image = image.mean(0, keepdim=True)
                    image = image.add(1).div(2)
                    TF.to_pil_image(image.clamp(0, 1)).save(filename)
                    tqdm.write(f'Batch {i}, step {j}, output {k}:')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    image = Image.open("./scripts/azimuth-color.jpeg").convert('RGB')
    image = np.array(image)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) / 255.0
    image = image * 2 - 1
    image = F.interpolate(
        image, size=(256, 256), mode="bilinear", align_corners=False
    )

    image = image.to(device)

    # image = image[..., 256//2 - 64: 256//2 + 64, 256//2 - 64: 256//2 + 64]

    # Load models

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load('./models/256x256_diffusion_uncond.pt', map_location='cpu'))
    model.requires_grad_(True).eval().to(device)
    if model_config['use_fp16']:
        model.convert_to_fp16()

    gc.collect()
    do_run(model, diffusion, image)
