from setuptools import setup, find_packages

setup(
    name="multimodal_search",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.30.0",
        "torchvision>=0.11.0",
        "pillow>=8.0.0",
        "numpy>=1.19.0",
        "evaluate>=0.4.0",
        "datasets>=2.10.0",
        "scikit-learn>=1.0.0",
        "wandb",
    ],
) 