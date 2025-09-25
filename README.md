# optimization tools for PHYsics based on Bayesian Optimization ( PHYSBO )

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

## Dependencies

- Python >= 3.9
- NumPy
- SciPy

### About Cython

From v3.0.0, PHYSBO no longer uses Cython in order to simplify installation process particularly on Windows.
This means that the performance of PHYSBO is slightly degraded from older versions.
If you need more performance, you can install ``physbo-core-cython`` additionally.
This package offers Cythonized version of some functions of PHYSBO.

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

- To install ``physbo-core-cython`` ::

  ```bash
  python3 -m pip install physbo-core-cython
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

## License

PHYSBO was developed based on [COMBO](https://github.com/tsudalab/COMBO) for academic use.
PHYSBO is distributed under Mozilla Public License version 2.0 (MPL v2).
We hope that you cite the following reference when you publish the results using PHYSBO:

[“Bayesian optimization package: PHYSBO”, Yuichi Motoyama, Ryo Tamura, Kazuyoshi Yoshimi, Kei Terayama, Tsuyoshi Ueno, Koji Tsuda, Computer Physics Communications Volume 278, September 2022, 108405.](https://doi.org/10.1016/j.cpc.2022.108405)

Bibtex

```bibtex
@misc{@article{MOTOYAMA2022108405,
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

### Copyright

© *2020- The University of Tokyo. All rights reserved.*
This software was developed with the support of \"*Project for advancement of software usability in materials science*\" of The Institute for Solid State Physics, The University of Tokyo.
