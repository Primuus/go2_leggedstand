from setuptools import find_namespace_packages, setup

setup(
    name="go2-hindstand-gym",
    version="0.1.0",
    description="Minimal upstream Go2 hind-leg standing and walking task",
    packages=find_namespace_packages(include=["legged_gym", "legged_gym.*"]),
    python_requires=">=3.8",
    install_requires=["isaacgym", "rsl-rl", "numpy==1.24.4", "scipy", "matplotlib", "tensorboard"],
)
