from setuptools import find_packages, setup

setup(
    name="egg",
    version="0.0.0",
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
    packages=find_packages(exclude=[]),
)
