from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("learnedSpectrum/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="learnedSpectrum",
    version="0.1.0",
    author="Teddy Warner",
    author_email="tawarner@usc.edu",
    description="fMRI Learning Stage Classification with Vision Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Twarner491/learnedSpectrum",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.9",
        ]
    },
    entry_points={
        "console_scripts": [
            "learnedspectrum=learnedSpectrum.scripts.train:main",
        ],
    },
    include_package_data=True,
    package_data={
        "learnedSpectrum": ["requirements.txt"],
    },
)
