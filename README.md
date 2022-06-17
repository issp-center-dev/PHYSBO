# optimization tools for PHYsics based on Bayesian Optimization ( PHYSBO )

Bayesian optimization has been proven as an effective tool in accelerating scientific discovery.
A standard implementation (e.g., scikit-learn), however, can accommodate only small training data.
PHYSBO is highly scalable due to an efficient protocol that employs Thompson sampling, random feature maps, one-rank Cholesky update and automatic hyperparameter tuning. Technical features are described in [COMBO's document](https://github.com/tsudalab/combo/blob/master/docs/combo_document.pdf) and [PHYSBO's report](https://doi.org/10.1016/j.cpc.2022.108405) (open access).
PHYSBO was developed based on [COMBO](https://github.com/tsudalab/combo) for academic use.

## Document

- Stable (master branch)
    - [English](https://issp-center-dev.github.io/PHYSBO/manual/master/en/index.html)
    - [日本語](https://issp-center-dev.github.io/PHYSBO/manual/master/ja/index.html)
- Latest (develop branch)
    - [English](https://issp-center.dev.github.io/PHYSBO/manual/develop/en/index.html)
    - [日本語](https://issp-center-dev.github.io/PHYSBO/manual/develop/ja/index.html)

## Required Packages

- Python >= 3.6
- numpy
- scipy

## Install

- From PyPI (recommended)

```bash
pip3 install physbo
```

- From source (for developers)
    1. Update pip (>= 19.0)

        ```bash
        pip3 install -U pip
        ```

    1. Download or clone the github repository

        ```
        git clone https://github.com/issp-center-dev/PHYSBO
        ```

    1. Install via pip
        ``` bash
        # ./PHYSBO is the root directory of PHYSBO
        # pip install options such as --user are avaiable

        pip3 install ./PHYSBO
        ```

    1. Note: Do not `import physbo` at the root directory of the repository because `import physbo` does not try to import the installed PHYSBO but one in the repository, which includes Cython codes not compiled.

## Uninstall

```bash
pip3 uninstall physbo
```

## Usage

['examples/simple.py'](https://github.com/issp-center-dev/PHYSBO/examples/simple.py) is a simple example.

## License

PHYSBO was developed based on [COMBO](https://github.com/tsudalab/COMBO) for academic use.
This package is distributed under GNU General Public License version 3 (GPL v3) or later.
We hope that you cite the following reference when you publish the results using PHYSBO:

[“Bayesian optimization package: PHYSBO”, Yuichi Motoyama, Ryo Tamura, Kazuyoshi Yoshimi, Kei Terayama, Tsuyoshi Ueno, Koji Tsuda, Computer Physics Communications Volume 278, September 2022, 108405.](https://doi.org/10.1016/j.cpc.2022.108405)

Bibtex

```
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
