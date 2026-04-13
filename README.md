> [!WARNING]
> This package has been migrated to the [TeamTomo monorepo](https://github.com/teamtomo/teamtomo).
> Future development, bug fixes, and releases will happen there.
> This repository is archived and no longer maintained.
> This package is still published to and installable from the same PyPI project, but development installations should be made from the monorepo.

# torch-fourier-filter

[![License](https://img.shields.io/pypi/l/torch-fourier-filter.svg?color=green)](https://github.com/jdickerson95/torch-fourier-filter/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-fourier-filter.svg?color=green)](https://pypi.org/project/torch-fourier-filter)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-fourier-filter.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/torch-fourier-filter/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/torch-fourier-filter/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/torch-fourier-filter/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/torch-fourier-filter)

Fourier space filters for image and volumes in pyTorch

Install via source using
```zsh
pip install -e .
```
And for development and testing use
```zsh
pip install -e ".[dev,test]"
```

Make sure to run tests before any commits:
```zsh
python -m pytest
```

