# Imports

import gc
import sys

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from torchvision.transforms import functional as TF
import torchvision.models as models

from tqdm import tqdm
from lib.nnvision.nnvision.models.trained_models.v4_task_driven import task_driven_ensemble_1, task_driven_ensemble_2
from lib.nnvision.nnvision.models.trained_models.v4_data_driven import \
    v4_multihead_attention_ensemble_model, v4_multihead_attention_ensemble_model_2

task_driven_ensemble_1.eval()
task_driven_ensemble_2.eval()

v4_multihead_attention_ensemble_model.eval()
v4_multihead_attention_ensemble_model_2.eval()

task_driven_ensemble_1.cuda()
task_driven_ensemble_2.cuda()

v4_multihead_attention_ensemble_model.cuda()
v4_multihead_attention_ensemble_model_2.cuda()

from PIL import Image
sys.path.append('./guided-diffusion')

from functools import partial

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from torchvision import transforms

images = torch.Tensor(np.load('./scripts/75_monkey_test_imgs.npy'))

# Model settings
model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '1000',  # Modify this value to decrease the number of timesteps.
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
energy_scale = 2 # 20
seed = 0
norm_constraint = 100 # 25
model_type = 'task_driven' # 'task_driven' or 'v4_multihead_attention'
# units = [995, 403, 1017, 67, 82, 88, 90, 99, 182, 227, 289, 315, 1123, 246, 789, 790, 831, 1068, 115, 649, 710, 546, 569, 589, 6, 602]
progressive = True
image = 0

def do_run(model, diffusion, energy_fn, desc='progress', grayscale=False):
    if seed is not None:
        torch.manual_seed(seed)

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    cur_t = diffusion.num_timesteps - skip_timesteps - 1

    samples = sample_fn(
        model,
        (batch_size, 3, model_config['image_size'], model_config['image_size']),
        clip_denoised=False,
        model_kwargs={},
        progress=True,
        energy_fn=energy_fn,
        energy_scale=energy_scale,
    )

    for j, sample in enumerate(samples):
        cur_t -= 1
        if (j % 10 == 0 and progressive) or cur_t == -1:
            print()

            energy = energy_fn(sample['pred_xstart'])

            for k, image in enumerate(sample['pred_xstart']):
                filename = f'{desc}_{0:05}.png'
                if grayscale:
                    image = image.mean(0, keepdim=True)
                image = image.add(1).div(2)

                image = image.clamp(0, 1)
                TF.to_pil_image(image).save(filename)

                tqdm.write(f'step {j} | train energy: {energy["train"]:.4g} | val energy: {energy["val"]:.4g} | cross-val energy: {energy["cross-val"]:.4g}')

    return energy


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    def energy_fn(x, target_response=None, val_response=None, cross_val_response=None, norm=100, models=None):
        tar = F.interpolate(x, size=(100, 100), mode='bilinear', align_corners=False).mean(1, keepdim=True)
        # normalize
        # tar = transforms.Normalize(
        #     [0.4876],
        #     [
        #         0.2756,
        #     ],
        # )(tar)

        tar = tar / torch.norm(tar) * norm  # 60
        tar = tar.clip(-1.7, 1.9)

        train_pred = models['train'](tar, data_key="all_sessions", multiplex=True)[0]
        val_pred = models['val'](tar, data_key="all_sessions", multiplex=True)[0]
        cross_val_pred = models['cross-val'](tar, data_key="all_sessions", multiplex=False)[0]

        train_energy = torch.mean((train_pred - target_response)**2)
        val_energy = torch.mean((val_pred - val_response)**2)
        cross_val_energy = torch.mean((cross_val_pred - cross_val_response)**2)

        return {
            'train': train_energy,
            'val': val_energy,
            'cross-val': cross_val_energy,
        }


    # model, diffusion = create_model_and_diffusion(**model_config)
    # model.load_state_dict(torch.load('./models/256x256_diffusion_uncond.pt', map_location='cpu'))
    # model.requires_grad_(True).eval().to(device)
    # if model_config['use_fp16']:
    #     model.convert_to_fp16()

    gc.collect()

    models = {
        'task_driven': {
            'train': task_driven_ensemble_1,
            'val': task_driven_ensemble_2,
            'cross-val': v4_multihead_attention_ensemble_model,
        },
        'v4_multihead_attention': {
            'train': v4_multihead_attention_ensemble_model,
            'val': v4_multihead_attention_ensemble_model_2,
            'cross-val': task_driven_ensemble_1,
        },
        'cross': {
            'train': task_driven_ensemble_1,
            'val': task_driven_ensemble_2,
            'cross-val': v4_multihead_attention_ensemble_model,
        }
    }

    train_scores = []
    val_scores = []
    cross_val_scores = []
    target_image = images[image].unsqueeze(0).unsqueeze(0).to(device)
    target_response = models[model_type]['train'](target_image, data_key="all_sessions", multiplex=True)[0]
    val_response = models[model_type]['val'](target_image, data_key="all_sessions", multiplex=True)[0]
    cross_val_response = models[model_type]['cross-val'](target_image, data_key="all_sessions", multiplex=False)[0]

    # optimize image to minimize energy using Adam
    class ImageGen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image = torch.nn.Parameter(torch.randn(1, 3, 256, 256).to(device))

        def forward(self):
            return self.image

    image_gen = ImageGen().to(device)
    optimizer = torch.optim.Adam(image_gen.parameters(), lr=0.1)


    pbar = tqdm(range(500))
    for i in pbar:
        image = image_gen()
        image = F.interpolate(image, size=(100, 100), mode='bilinear', align_corners=False).mean(1, keepdim=True)
        image = image / torch.norm(image) * 100
        # image = image.clip(-1.7, 1.9)

        res = models[model_type]['train'](image, data_key="all_sessions", multiplex=True)[0]
        val_res = models[model_type]['val'](image, data_key="all_sessions", multiplex=True)[0]
        cross_val_res = models[model_type]['cross-val'](image, data_key="all_sessions", multiplex=False)[0]

        loss = torch.mean((res - target_response)**2)
        val_loss = torch.mean((val_res - val_response)**2)
        cross_loss = torch.mean((cross_val_res - cross_val_response)**2)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(f'loss: {loss.item()} | val_loss: {val_loss.item()} | cross_loss: {cross_loss.item()}')


    # save image
    image = image.detach().cpu().numpy()[0, 0]
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(f'./{model_type}_{norm_constraint}.png')


    # target = np.load('cute_monkey_60k_responses.npy')[0, ::49]
    # target = torch.from_numpy(target).float().cuda()

    # score = do_run(
    #     model=model,
    #     diffusion=diffusion,
    #     energy_fn=partial(energy_fn, target_response=target_response, val_response=val_response, cross_val_response=cross_val_response, norm=norm_constraint, models=models[model_type]),
    #     desc=f'progress_{image}',
    #     grayscale=True
    # )
    # train_scores.append(score['train'].item())
    # val_scores.append(score['val'].item())
    # cross_val_scores.append(score['cross-val'].item())

    # print('Train:', train_scores)
    # print('Val:', val_scores)
    # print('Cross-val:', cross_val_scores)
