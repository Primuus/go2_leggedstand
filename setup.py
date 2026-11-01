from setuptools import find_namespace_packages, setup

setup(
    name="go2-hindstand-gym",
    version="0.1.0",
    description="Reference Go2 hindstand task with an Isaac Lab backend",
    packages=find_namespace_packages(include=["go2_hindstand_lab", "go2_hindstand_lab.*", "deploy_mujoco", "legged_gym", "legged_gym.*"]),
    python_requires=">=3.11",
    # Isaac Lab is provided by the server; ry_python.sh selects the bundled PPO.
    install_requires=["numpy>=1.24", "scipy", "torch", "matplotlib", "tensorboard"],
)
