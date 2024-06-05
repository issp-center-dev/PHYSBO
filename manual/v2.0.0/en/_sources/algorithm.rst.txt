.. _chap_algorithm:

Algorithm
=====================
This section describes an overview of Bayesian optimization. For technical details, please refer to `this reference  <https://github.com/tsudalab/combo/blob/master/docs/combo_document.pdf>`_ .

Bayesian optimization
---------------------
Bayesian optimization is a method that can be used in complex simulations or real-world experimental tasks where the evaluation of the objective function (e.g., property values) is very costly. In other words, Bayesian optimization solves the problem of finding explanatory variables (material composition, structure, process and simulation parameters, etc.) that have a better objective function (material properties, etc.) with as few experiments and simulations as possible. In Bayesian optimization, we start from a situation where we have a list of candidates for the explanatory variables to be searched (represented by the vector :math:`{\bf x}`). Then, from among the candidates, the one that is expected to improve the objective function :math:`y` is selected by making good use of prediction by machine learning (using Gaussian process regression). We then evaluate the value of the objective function by performing experiments and simulations on the candidates. By repeating the process of selection by machine learning and evaluation by experimental simulation, optimization can be achieved in as few times as possible.

The details of the Bayesian optimization algorithm are described below.

- Step1: Initialization

Prepare the space to be explored in advance. In other words, list up the composition, structure, process, simulation parameters, etc. of the candidate materials as a vector :math:`{\bf x}`. At this stage, the value of the objective function is not known. A few candidates are chosen as initial conditions and  the value of the objective function :math:`y` is estimated by experiment or simulation. This gives us the training data :math:`D = \{ {\bf x}_i, y_i \}_{(i=1, \cdots, N)}` with the explanatory variables :math:`{\bf x}` and the objective function :math:`y`.

- Step2: Selection of candidates

Using the training data, learn a Gaussian process. For Gaussian process, the mean of the predictions at arbitary :math:`{\bf x}` is :math:`\mu_c ({\bf x})` and the variance is :math:`\sigma_c ({\bf x})` are given as follows

.. math::
   
   \mu_c ({\bf x}) &= {\bf k}({\bf x})^T (K+\sigma^2 I)^{-1}{\bf y},

   \sigma_c({\bf x}) &= k({\bf x}, {\bf x}) + \sigma^2 - {\bf k}({\bf x})^T  (K+\sigma^2 I)^{-1}{\bf k}({\bf x}),

where :math:`k({\bf x}, {\bf x}')` is a function called as a kernel, and it represents the similarity of two vectors. In general, the following Gaussian kernel is used:

.. math::

   k({\bf x}, {\bf x}') = \exp \left[ -\frac{1}{2\eta^2}||{\bf x} - {\bf x}'||^2 \right].

Using this kernel function, :math:`{\bf k}({\bf x})` and :math:`K` are computed as follows

.. math::
   
   {\bf k}({\bf x}) = \left( k({\bf x}_1, {\bf x}), k({\bf x}_2, {\bf x}), \cdots, k({\bf x}_N, {\bf x}) \right)^\top

.. math::
   :nowrap:

    \[
    K = \left(
    \begin{array}{cccc}
       k({\bf x}_1, {\bf x}_1) & k({\bf x}_1, {\bf x}_2) & \ldots &  k({\bf x}_1, {\bf x}_N) \\
       k({\bf x}_2, {\bf x}_1) & k({\bf x}_2, {\bf x}_2) & \ldots &  k({\bf x}_2, {\bf x}_N) \\
      \vdots & \vdots & \ddots & \vdots \\
       k({\bf x}_N, {\bf x}_1) & k({\bf x}_N, {\bf x}_2) & \ldots &  k({\bf x}_N, {\bf x}_N)
    \end{array}
    \right)
    \]

For all candidates that have not yet been tested or simulated, the prediction :math:`\mu_c ({\bf x})` and the variance associated with the uncertainty of the prediction :math:`\sigma_c ({\bf x})` are estimated. Using this, the acquisition function is calculated. Then, the candidate :math:`{\bf x}^*` is selected that maximizes the acquisition function from among the candidates for which we do not yet know the value of the objective function. In this case, :math:`\sigma` and :math:`\eta` are called hyperparameters, and PHYSBO will automatically set the best value.

As an acquisition function, for example, Maximum Probability of Improvement (PI) and Maximum Expected Improvement (EI) are useful.
The score of PI is defined as follows.

.. math::

   \text{PI} (\mathbf{x}) = \Phi (z (\mathbf{x})), \ \ \ z(\mathbf{x}) = \frac{\mu_c (\mathbf{x}) - y_{\max}}{\sigma_c (\mathbf{x})},
   
where :math:`\Phi(\cdot)` is the cumulative distribution function.
The PI score represents the probability of exceeding the maximum :math:`y_{\max}` of the currently obtained :math:`y`.
In addition, the EI score is the expected value of the difference between the predicted value and the current maximum :math:`y_{\max}` and is given by

.. math::

   \text{EI} (\mathbf{x}) = [\mu_c (\mathbf{x})-y_{\max}] \Phi (z (\mathbf{x})) + \sigma_c (\mathbf{x}) \phi (z (\mathbf{x})), \ \ \ z(\mathbf{x}) = \frac{\mu_c (\mathbf{x}) - y_{\max}}{\sigma_c (\mathbf{x})},

where :math:`\phi(\cdot)` is a probability density function.


- Step3: Experiment (Simulation)

Perform an experiment or simulation on the candidate :math:`{\bf x}^*` with the largest acquisition function selected in step 2, and estimate the objective function value :math:`y`. This will add one more piece of training data. Repeat steps 2 and 3 to search for candidates with good scores.

Accelerating Bayesian Optimization with PHYSBO
-----------------------------------------------

In PHYSBO, random feature map, Thompson sampling, and Cholesky decomposition are used to accelerate the calculation of Bayesian optimization.
First, the random feature map is introduced.
By introducing the random feature map :math:`\phi (\mathbf{x})`, we can approximate the Gaussian kernel :math:`k(\mathbf{x},\mathbf{x}')` as follows.

.. math::

   k(\mathbf{x},\mathbf{x}') = \exp \left[ - \frac{1}{2 \eta^2} \| \mathbf{x} -\mathbf{x}' \| \right]^2  \simeq \phi (\mathbf{x})^\top \phi(\mathbf{x}') \\
   \phi (\mathbf{x}) = \left( z_{\omega_1, b_1} (\mathbf{x}/\eta),..., z_{\omega_l, b_l} (\mathbf{x}/\eta) \right)^\top,

where :math:`z_{\omega, b} (\mathbf{x}) = \sqrt{2} \cos (\boldsymbol{\omega}^\top \mathbf{x}+b)`.
Then, :math:`\boldsymbol{\omega}` is generated from :math:`p(\boldsymbol{\omega}) = (2\pi)^{-d/2} \exp (-\|\boldsymbol{\omega}\|^2/2)` and :math:`b` is chosen uniformly from :math:`[0, 2 \pi]` is chosen uniformly from :math:`[0, 2 \pi]`.
This approximation is strictly valid in the limit of :math:`l \to \infty`, where the value of :math:`l` is the dimension of the random feature map.

 :math:`\Phi` can be represented as a :math:`l` row :math:`n` column matrix with :math:`\phi(\mathbf{x}_i)` in each column by :math:`\mathbf{x}` vector of training data as follows:

.. math::

   \Phi = ( \phi(\mathbf{x}_1),..., \phi(\mathbf{x}_n) ).

It is seen that the following relation is satisfied:

.. math::

   \mathbf{k} (\mathbf{x}) = \Phi^\top \phi(\mathbf{x}) \\
   K= \Phi^\top \Phi.

Next, a method that uses Thompson sampling to make the computation time for candidate prediction :math:`O(l)` is introduced.
Note that using EI or PI will result in :math:`O(l^2)` because of the need to evaluate the variance.
In order to perform Thompson sampling, the Bayesian linear model defined below is used.

.. math::

   y = \mathbf{w}^\top \phi (\mathbf{x}),

where :math:`\phi(\mathbf{x})` is random feature map described above and :math:`\mathbf{w}` is a coefficient vector.
In a Gaussian process, when the training data :math:`D` is given, this :math:`\mathbf{w}` is determined to follow the following Gaussian distribution.

.. math::

   p(\mathbf{w}|D) = \mathcal{N} (\boldsymbol{\mu}, \Sigma) \\
   \boldsymbol{\mu} = (\Phi \Phi^\top + \sigma^2 I)^{-1} \Phi \mathbf{y} \\
   \Sigma = \sigma^2 (\Phi \Phi^\top + \sigma^2 I)^{-1}

In Thompson sampling, one coefficient vector is sampled according to this posterior probability distribution and set to :math:`\mathbf{w}^*`, which represents the acquisition function as follows

.. math::

   \text{TS} (\mathbf{x}) = {\mathbf{w}^*}^\top \phi (\mathbf{x}).

The :math:`\mathbf{x}^*` that maximizes :math:`\text{TS} (\mathbf{x})`  will be selected as the next candidate.
In this case, :math:`\phi (\mathbf{x})` is an :math:`l` dimensional vector, so the acquisition function can be computed with :math:`O(l)`.

Next, the manner for accelerating the sampling of :math:`\mathbf{w}` is introduced.
The matrix :math:`A` is defined as follows.

.. math::

   A = \frac{1}{\sigma^2} \Phi \Phi^\top +I

Then the posterior probability distribution is given as

.. math::

   p(\mathbf{w}|D) = \mathcal{N} \left( \frac{1}{\sigma^2} A^{-1} \Phi \mathbf{y}, A^{-1} \right).

Therefore, in order to sample :math:`\mathbf{w}`, we need to calculate :math:`A^{-1}`.
Now consider the case of the newly added :math:`(\mathbf{x}', y')` in the Bayesian optimization iteration.
With the addition of this data, the matrix :math:`A` is updated as

.. math::

   A' = A + \frac{1}{\sigma^2} \phi (\mathbf{x}') \phi (\mathbf{x}')^\top.

This update can be done using the Cholesky decomposition ( :math:`A= L^\top L` ), which reduces the time it takes to compute :math:`A^{-1}` to :math:`O(l^2)`.
If we compute :math:`A^{-1}` at every step, the numerical cost becomes :math:`O(l^3)`.
The :math:`\mathbf{w}` is obtained by 

.. math::

   \mathbf{w}^* = \boldsymbol{\mu} + \mathbf{w}_0,

where  :math:`\mathbf{w}_0` is sampled from :math:`\mathcal{N} (0,A^{-1})` and :math:`\boldsymbol{\mu}` is calculated by

.. math::

   L^\top L \boldsymbol{\mu} = \frac{1}{\sigma^2} \Phi \mathbf{y}.

By using these techniques, a computation time becomes almost linear in the number of training data.
