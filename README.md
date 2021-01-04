# optimization tools for PHYsics based on Bayesian Optimization ( PHYSBO )
Bayesian optimization has been proven as an effective tool in accelerating scientific discovery.
A standard implementation (e.g., scikit-learn), however, can accommodate only small training data.
PHYSBO is highly scalable due to an efficient protocol that employs Thompson sampling, random feature maps, one-rank Cholesky update and automatic hyperparameter tuning. Technical features are described in [COMBO's document](https://github.com/tsudalab/combo/blob/master/docs/combo_document.pdf).
PHYSBO was developed based on [COMBO](https://github.com/tsudalab/combo) for academic use.

# Document #####################################

- english (in preparation)
- [日本語](https://issp-center-dev.github.io/PHYSBO/manual/master/ja/index.html)

# Required Packages ############################
* Python >= 3.6
* numpy
* scipy

# Install ######################################
- From PyPI (recommended)
```bash
    $ pip3 install physbo
```

- From source (for developers)
    1. Update pip (>= 19.0)
        ```bash
        $ pip3 install -U pip
        ```

    1. Download or clone the github repository
        ```
        $ git clone https://github.com/issp-center-dev/PHYSBO
        ```

    1. Edit some files

    1. Install via pip
        ``` bash
        # ./PHYSBO is the root directory of PHYSBO
        # pip install options such as --user are avaiable

        $ pip3 install ./PHYSBO
        ```

    1. Note: Do not `import physbo` at the root directory of the repository because `import physbo` does not try to import the installed PHYSBO but one in the repository, which includes Cython codes not compiled.

# Uninstall
```bash
$ pip3 uninstall physbo
```

# Usage
After installation, you can launch the test suite from ['examples/grain_bound/tutorial.ipynb'](https://github.com/issp-center-dev/PHYSBO/examples/grain_bound/tutorial.ipynb).

## License
PHYSBO was developed based on [COMBO](https://github.com/tsudalab/COMBO) for academic use.
This package is distributed under GNU General Public License version 3 (GPL v3) or later.

Copyright
---------

© *2020- The University of Tokyo. All rights reserved.*
This software was developed with the support of \"*Project for advancement of software usability in materials science*\" of The Institute for Solid State Physics, The University of Tokyo. 
