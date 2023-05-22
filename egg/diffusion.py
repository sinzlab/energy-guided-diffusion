import torch
import torch.nn as nn
from egg.guided_diffusion.script_util import (create_model_and_diffusion,
                                          model_and_diffusion_defaults)

class EGG(nn.Module):
    def __init__(self, diffusion_artefact="./models/256x256_diffusion_uncond.pt", config=None, num_steps=50):
        # Model settings
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": False,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": num_steps,
                "image_size": 256,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_checkpoint": False,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )

        if config is not None:
            self.model_config.update(config)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(
            torch.load(diffusion_artefact, map_location="cpu")
        )
        self.model.requires_grad_(True).eval().to(device)
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

    def sample(self, energy_fn, energy_scale=1, num_samples=1):
        return self.diffusion.p_sample_loop_progressive(
            self.model,
            (num_samples, 3, self.model_config["image_size"], self.model_config["image_size"]),
            clip_denoised=False,
            model_kwargs={},
            progress=True,
            energy_fn=energy_fn,
            energy_scale=energy_scale,
        )


