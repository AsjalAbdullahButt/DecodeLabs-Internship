from setuptools import setup, find_packages

setup(
    name="project4-recognition",
    version="1.0.0",
    author="Asjal Abdullah Butt",
    description="Image and Text Recognition Pipeline — DecodeLabs Project 4",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=open("requirements.txt").read().splitlines(),
)
