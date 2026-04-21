from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="binningverdict",
    version="1.0.0",
    description="Log-vs-linear binning decision system for SANS data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="[Lead Author]",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy>=1.20"],
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
