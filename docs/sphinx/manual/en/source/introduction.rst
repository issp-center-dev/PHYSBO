Introduction
=====================

About PHYSBO
----------------------

PHYSBO (optimization tools for PHYSics based on Bayesian Optimization) is a Python library for fast and scalable Bayesian optimization. It is based on COMBO (Common Bayesian Optimization) and has been developed mainly for researchers in the materials science field. There are many attempts to accelerate scientific discovery through data-driven design-of-experiment algorithms in the fields of physics, chemistry, and materials. Bayesian optimization is an effective tool for accelerating these scientific discoveries. Bayesian optimization is a technique that can be used for complex simulations and real-world experimental tasks where the evaluation of objective function values (e.g., characteristic values) is very costly. In other words, the problem solved by Bayesian optimization is to find a parameter (e.g., material composition, structure, process and simulation parameters) with a better objective function value (e.g., material properties) in as few experiments and simulations as possible. In Bayesian optimization, the candidate parameters to be searched for are listed in advance, and the candidate with the largest objective function value is selected from among the candidates by making good use of machine learning (using Gaussian process regression) prediction. Experiments and simulations are performed on the candidates and the objective function values are evaluated. By repeating the process of selection by machine learning and evaluation by experimental simulation, we can reduce the number of times of optimization. On the other hand, Bayesian optimization is generally computationally expensive, and standard implementations such as scikit-learn are difficult to handle a large amount of data. PHYSBO achieves high scalability due to the following features

- Thompson Sampling
- random feature map
- one-rank Cholesky update
- automatic hyperparameter tuning

Please see `this reference <https://github.com/tsudalab/combo/blob/master/docs/combo_document.pdf>`_ for technical details.

Citation
----------------------

When citing PHYSBO, please cite the following reference:

Tsuyoshi Ueno, Trevor David Rhone, Zhufeng Hou, Teruyasu Mizoguchi and Koji Tsuda,
COMBO: An Efficient Bayesian Optimization Library for Materials Science,
Materials Discovery 4, 18-21 (2016). Available from https://doi.org/10.1016/j.md.2016.04.001

Bibtex is given as follows::

    @article{Ueno2016,
    title = "COMBO: An Efficient Bayesian Optimization Library for Materials Science ",
    journal = "Materials Discovery",
    volume = "4",
    pages = "18-21",
    year = "2016",
    doi = "http://dx.doi.org/10.1016/j.md.2016.04.001",
    url = "http://www.sciencedirect.com/science/article/pii/S2352924516300035",
    author = "Tsuyoshi Ueno and Trevor David Rhone and Zhufeng Hou and Teruyasu Mizoguchi and Koji Tsuda",
    }

Main Developers
----------------------

- ver. 1.0-

  - Ryo Tamura (International Center for Materials Nanoarchitectonics, National Institute for Materials Science)
  - Tsuyoshi Ueno (Magne-Max Capital Management Company)
  - Kei Terayama (Graduate School of Medical Life Science, Yokohama City University)
  - Koji Tsuda (Graduate School of Frontier Sciences, The University of Tokyo)
  - Yuichi Motoyama (The Institute for Solid State Physics, The University of Tokyo)
  - Kazuyoshi Yoshimi (The Institute for Solid State Physics, The University of Tokyo)
  - Naoki Kawashima (The Institute for Solid State Physics, The University of Tokyo)

License
----------------------

GNU General Public License version 3

Copyright (c) <2020-> The University of Tokyo. All rights reserved.

Part of this software is developed under the support of “Project for advancement of software usability in materials science” by The Institute for Solid State Physics, The University of Tokyo.
