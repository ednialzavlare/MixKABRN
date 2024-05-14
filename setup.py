from setuptools import setup, find_packages

setup(
    name="my_mixkabrn_model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
)

