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
from lib.nnvision.nnvision.models.trained_models.v4_task_driven import task_driven_ensemble_1

from PIL import Image

from functools import partial

sys.path.append('./guided-diffusion')

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

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
energy_scale = 1.0
seed = 0
norm_constraint = 100

style_w = 1_000
content_w = 1


def do_run(model, diffusion, target_image, energy_fn):
    if seed is not None:
        torch.manual_seed(seed)

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
            energy_scale=energy_scale,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % 10 == 0 or cur_t == -1:
                print()
                for k, image in enumerate(sample['pred_xstart']):
                    filename = f'progress_{0:05}.png'
                    # image = image.mean(0, keepdim=True)
                    image = image.add(1).div(2)
                    image = image.clamp(0, 1)
                    TF.to_pil_image(image).save(filename)
                    tqdm.write(f'Batch {i}, step {j}, output {k}:')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    content_img = Image.open("./scripts/dancing.jpg").convert('RGB')
    content_img = np.array(content_img)
    content_img = torch.from_numpy(content_img).permute(2, 0, 1).unsqueeze(0) / 255.0
    content_img = content_img * 2 - 1
    content_img = F.interpolate(
        content_img, size=(256, 256), mode="bilinear", align_corners=False
    )
    content_img = content_img.to(device)

    style_img = Image.open("./scripts/picasso.jpg").convert('RGB')
    style_img = np.array(style_img)
    style_img = torch.from_numpy(style_img).permute(2, 0, 1).unsqueeze(0) / 255.0
    style_img = style_img * 2 - 1
    style_img = F.interpolate(
        style_img, size=(256, 256), mode="bilinear", align_corners=False
    )
    style_img = style_img.to(device)


    class ForwardHook():
        def __init__(self, module):
            self.hook = module.register_forward_hook(self.hook_fn)

        def hook_fn(self, module, input, output):
            self.input = input
            self.output = output

        def close(self):
            self.hook.remove()


    def per_layer_features(x, model, content_layers, style_layers, transform=None):
        out = transform(x) if transform is not None else x
        content_hooks = [ForwardHook(layer) for layer in content_layers]
        style_hooks = [ForwardHook(layer) for layer in style_layers]

        model_output = model(out)
        content_outputs = [hook.output for hook in content_hooks]
        style_outputs = [hook.output for hook in style_hooks]
        return {"content": content_outputs, "style": style_outputs}


    def content_loss(output, target):
        return F.mse_loss(output, target.detach())

    def gram_matrix(featmap):
        b, c, h, w = featmap.size()
        features = featmap.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)

    def style_loss(output, target):
        return F.mse_loss(gram_matrix(output), gram_matrix(target).detach())


    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    model = nn.Sequential()
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

    cnn = model

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


    # create a module to normalize input image so we can easily put it in a
    # nn.Sequential
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std


    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    transform = Normalization(normalization_mean, normalization_std)

    content_layers = [cnn.__getattr__(layer) for layer in content_layers]
    style_layers = [cnn.__getattr__(layer) for layer in style_layers]
    target = per_layer_features(content_img,
                                model=cnn,
                                content_layers=content_layers,
                                style_layers=style_layers,
                                transform=transform)

    content_target = target["content"]

    target = per_layer_features(style_img,
                                model=cnn,
                                content_layers=content_layers,
                                style_layers=style_layers,
                                transform=transform)
    style_target = target["style"]

    # target = np.load('cute_monkey_60k_responses.npy')
    # target = torch.from_numpy(target).float().cuda()

    def energy_fn(x):
        # x = x / torch.norm(x) * norm_constraint
        output = per_layer_features(x,
                                    model=cnn,
                                    content_layers=content_layers,
                                    style_layers=style_layers,
                                    transform=transform)

        style_score, content_score = 0, 0
        for style_out, style_t in zip(output["style"], style_target):
            style_score += style_loss(style_out, style_t)
        for content_out, content_t in zip(output["content"], content_target):
            content_score += content_loss(content_out, content_t)

        print('style_score', style_score.item(), 'content_score', content_score.item())

        return style_w * style_score + content_w * content_score

        # tar = F.interpolate(x.clone(), size=(100, 100), mode='bilinear', align_corners=False).mean(1, keepdim=True)
        #
        # # normalize
        # tar = tar / torch.norm(tar) * norm_constraint  # 60
        #
        # response = task_driven_ensemble_1(tar, data_key="all_sessions", multiplex=True)[0]
        #
        # return torch.mean((response[::49] - target[0, ::49]) ** 2)

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load('./models/256x256_diffusion_uncond.pt', map_location='cpu'))
    model.requires_grad_(True).eval().to(device)
    if model_config['use_fp16']:
        model.convert_to_fp16()

    gc.collect()
    do_run(model, diffusion, content_img, energy_fn)
