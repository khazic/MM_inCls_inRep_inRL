from setuptools import setup, find_packages

setup(
    name="MM_inCls_inRep_inRL",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.6.0",
        "transformers==4.49.0",
        "torchvision==0.21.0",
        "pillow==11.1.0",
        "numpy==2.2.4",
        "evaluate==0.4.3",
        "datasets==3.4.0",
        "scikit-learn==1.6.1",
        "wandb==0.19.8",
        "joblib==1.4.2",
        "tqdm==4.67.1",
        "flash-attn==2.7.1.post4",
    ],
)