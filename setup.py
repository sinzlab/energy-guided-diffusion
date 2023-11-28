from setuptools import find_packages, setup

setup(
    name="egg",
    version="0.0.1",
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "wandb"],
    packages=find_packages(),
    description="Image generation via energy guided diffusion for neural data",
)
