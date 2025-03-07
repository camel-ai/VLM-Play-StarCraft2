from setuptools import setup, find_packages

setup(
    name='vlm_attention',
    version='0.1.0',
    description='VLM Attention for StarCraft II',
    packages=find_packages(),
    # 不包含任何依赖，因为它们已经安装好了
)