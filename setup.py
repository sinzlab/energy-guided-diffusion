from setuptools import setup

setup(
    name="energy-guided-diffusion",
    py_modules=["egg"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
