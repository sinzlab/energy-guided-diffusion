from setuptools import setup

setup(
    name="egg",
    py_modules=["egg"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
