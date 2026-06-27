<p align="center">
  <img src="https://raw.githubusercontent.com/issp-center-dev/PHYSBO/master/docs/sphinx/manual/_static/logo.png" alt="PHYSBO logo" width="360"><br>
  <b>optimization tools for PHYsics based on Bayesian Optimization</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/physbo/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/physbo.svg"></a>
  <a href="https://pypi.org/project/physbo/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/physbo.svg"></a>
  <a href="https://mozilla.org/MPL/2.0/"><img alt="License: MPL-2.0" src="https://img.shields.io/badge/license-MPL--2.0-brightgreen.svg"></a>
  <a href="https://github.com/issp-center-dev/PHYSBO/actions/workflows/python_package.yml"><img alt="Test python package" src="https://github.com/issp-center-dev/PHYSBO/actions/workflows/python_package.yml/badge.svg"></a>
  <a href="https://issp-center-dev.github.io/PHYSBO/"><img alt="Documentation" src="https://img.shields.io/badge/docs-online-blue.svg"></a>
  <a href="https://doi.org/10.1016/j.cpc.2022.108405"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2022.108405-blue.svg"></a>
</p>

Bayesian optimization has been proven as an effective tool in accelerating scientific discovery.
A standard implementation (e.g., scikit-learn), however, can accommodate only small training data.
PHYSBO is highly scalable due to an efficient protocol that employs Thompson sampling, random feature maps, one-rank Cholesky update and automatic hyperparameter tuning. Technical features are described in [COMBO's document](https://github.com/tsudalab/combo/blob/master/docs/combo_document.pdf) and [PHYSBO's report](https://doi.org/10.1016/j.cpc.2022.108405) (open access).
PHYSBO was developed based on [COMBO](https://github.com/tsudalab/combo) for academic use.

## Documentation

- Stable (master branch)
  - [English](https://issp-center-dev.github.io/PHYSBO/manual/master/en/index.html)
  - [日本語](https://issp-center-dev.github.io/PHYSBO/manual/master/ja/index.html)
- Latest (develop branch)
  - [English](https://issp-center-dev.github.io/PHYSBO/manual/develop/en/index.html)
  - [日本語](https://issp-center-dev.github.io/PHYSBO/manual/develop/ja/index.html)
- API Reference
  - [English](https://issp-center-dev.github.io/PHYSBO/manual/master/en/api.html)
  - [日本語](https://issp-center-dev.github.io/PHYSBO/manual/master/ja/api.html)

## Dependencies

- Python >= 3.9
- NumPy
- SciPy

### Optional dependencies

- [ODAT-SE](https://github.com/issp-center-dev/ODAT-SE) (for continuous space optimization)

  ``` bash
  python3 -m pip install odat-se
  ```

- In order to run examples/simple_time.py, matplotlib is required:
  ``` bash
  python3 -m pip install matplotlib
  ```

## Install

- From PyPI (recommended)

  ```bash
  python3 -m pip install physbo
  ```

- From source (for developers)

    1. Download or clone the github repository

        ```bash
        git clone https://github.com/issp-center-dev/PHYSBO
        ```

    1. Install via pip

        ``` bash
        # ./PHYSBO is the root directory of PHYSBO
        # pip install options such as --user are avaiable

        python3 -m pip install ./PHYSBO
        ```

## Uninstall

```bash
python3 -m pip uninstall physbo
```

## Usage

For an introductory tutorial please consult the documentation. ([English](https://issp-center-dev.github.io/PHYSBO/manual/master/en/notebook/tutorial_basic.html) / [日本語](https://issp-center-dev.github.io/PHYSBO/manual/develop/ja/install.html#id2))

['examples/simple.py'](./examples/simple.py) is a simple example.

## Data repository

A tutorial and a dataset of a paper about PHYSBO can be found in [PHYSBO Gallery](https://isspns-gitlab.issp.u-tokyo.ac.jp/physbo-dev/physbo-gallery).

## For developers

[UV](https://docs.astral.sh/uv/) is recommended to make a virtual environment and install dependencies for development.

### Run tests

When using UV, the command `uv sync --extra tests` will install the dependencies to run tests like `pytest`.
Test files are in `tests/` directory, and use `pytest` to run tests.

``` bash
uv run pytest tests
```

### Build documentation

When using UV, the command `uv sync --extra docs` will install the dependencies for building the documentation like `sphinx`.

The command

```bash
uv run bash docs/make_docs.sh
```

will build the documentation into `docs/built`.

## License

PHYSBO was developed based on [COMBO](https://github.com/tsudalab/COMBO) for academic use.
PHYSBO is distributed under Mozilla Public License version 2.0 (MPL v2).

We hope that you cite the following references when you publish the results using PHYSBO:

- ["Bayesian optimization package: PHYSBO", Yuichi Motoyama, Ryo Tamura, Kazuyoshi Yoshimi, Kei Terayama, Tsuyoshi Ueno, Koji Tsuda, Computer Physics Communications Volume 278, September 2022, 108405.](https://doi.org/10.1016/j.cpc.2022.108405)

  - Bibtex

  ```bibtex
  @article{PHYSBO-paper2022,
  title = {Bayesian optimization package: PHYSBO},
  journal = {Computer Physics Communications},
  volume = {278},
  pages = {108405},
  year = {2022},
  issn = {0010-4655},
  doi = {https://doi.org/10.1016/j.cpc.2022.108405},
  author = {Yuichi Motoyama and Ryo Tamura and Kazuyoshi Yoshimi and Kei Terayama and Tsuyoshi Ueno and Koji Tsuda},
  keywords = {Bayesian optimization, Multi-objective optimization, Materials screening, Effective model estimation}
  }
  ```

- ["Update of PHYSBO: Improving Usability and Portability of Bayesian Optimization for Physics and Materials Research", Yuichi Motoyama, Kazuyoshi Yoshimi, Tatsumi Aoyama, Kei Terayama, Koji Tsuda, and Ryo Tamura, arXiv:2603.01349](https://arxiv.org/abs/2603.01349)

  - Bibtex

  ```bibtex
  @misc{PHYSBO-paper2026,
  doi = {10.48550/ARXIV.2603.01349},
  url = {https://arxiv.org/abs/2603.01349},
  author = {Motoyama,  Yuichi and Yoshimi,  Kazuyoshi and Aoyama,  Tatsumi and Terayama,  Kei and Tsuda,  Koji and Tamura,  Ryo},
  keywords = {Computational Physics (physics.comp-ph),  Materials Science (cond-mat.mtrl-sci),  FOS: Physical sciences,  FOS: Physical sciences},
  title = {Update of PHYSBO: Improving Usability and Portability of Bayesian Optimization for Physics and Materials Research},
  publisher = {arXiv},
  year = {2026},
  copyright = {arXiv.org perpetual,  non-exclusive license}
  }
  ```

### Copyright

© *2020- The University of Tokyo. All rights reserved.*
This software was developed with the support of \"*Project for advancement of software usability in materials science*\" of The Institute for Solid State Physics, The University of Tokyo.
