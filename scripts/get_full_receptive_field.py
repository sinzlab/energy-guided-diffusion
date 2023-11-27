from functools import partial

import numpy as np
import torch
import torchvision

from egg.models import models

model = models["task_driven"]["train"]
# model = torch.nn.DataParallel(model).cuda()


class Model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, data_key="all_sessions")#[..., 403]


model = Model(model).cuda()

ground_truth = torch.Tensor(np.load('./data/75_monkey_test_imgs.npy')).cuda().unsqueeze(1).requires_grad_(True)

grad = torch.zeros(100, 100)
for i in range(75):
    image = ground_truth[[i]]
    responses = model(image).sum()
    grad += torch.autograd.grad(responses, image, retain_graph=True)[0].squeeze().detach().cpu().abs()

grad = grad.detach().cpu().numpy()

# gaussian blur
from scipy.ndimage import gaussian_filter
grad = gaussian_filter(grad, sigma=3)

# normalize to [0, 1]
grad = (grad - grad.min()) / (grad.max() - grad.min())

# anything above 0.5 is important so set to 1
grad[grad > 0.25] = 1

# upscale to 256x256 using torchvision
grad = torchvision.transforms.Resize((256, 256))(torch.Tensor(grad).unsqueeze(0).unsqueeze(0)).squeeze().numpy()

# smooth again
grad = gaussian_filter(grad, sigma=3)

np.save('full_mask.npy', grad)

import matplotlib.pyplot as plt
plt.imshow(grad)
plt.savefig('grad.png')
