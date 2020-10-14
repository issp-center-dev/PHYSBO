PHYsics Bayesian Optimization Library ( PHYSBO )
========
Bayesian optimization has been proven as an effective tool in accelerating scientific discovery.
A standard implementation (e.g., scikit-learn), however,
can accommodate only small training data.
PHYSBO was developed based on COMBO for academic use.
PHYSBO is highly scalable due to an efficient protocol that employs
Thompson sampling, random feature maps, one-rank Cholesky update and
automatic hyperparameter tuning. Technical features are described in [COMBO's document](/docs/combo_document.pdf).

# Required Packages ############################
* Python 2.7.x
* numpy  >=1.10
* scipy  >= 0.16
* Cython >= 0.22.1
* mpi4py >= 2.0 (optional)


# Install ######################################
	1. Download or clone the github repository, e.g.
		> git clone https://github.com/tsudalab/combo.git

	2. Run setup.py install
		> cd combo
		> python setup.py install

# Uninstall

	1. Delete all installed files, e.g.
		> python setup.py install --record file.txt
		> cat file.txt  | xargs rm -rvf


# Usage
After installation, you can launch the test suite from ['examples/grain_bound/tutorial.ipynb'](examples/grain_bound/tutorial.ipynb).

## License

PHYSBO was developed based on COMBO for academic use.
This package is distributed under GNU General Public License version 3 (GPL v3) or later.

Copyright
---------

© *2020- The University of Tokyo. All rights reserved.*
This software was developed with the support of \"*Project for advancement of software usability in materials science*\" of The Institute for Solid State Physics, The University of Tokyo. 
