def call_simulator(simu, action):
    """

    Parameters
    ----------
    simu: simulator object
        This object is called in call_simulator and must have __call__(action).
    action: int
        Index of actions
    Returns
    -------
        t:  float
            The negative energy of the target candidate (value of the objective function to be optimized).
        X:  numpy array
            d dimensional matrix. The d-dimensional feature vector of the target candidate.
    """
    output = simu(action)
    if hasattr(output, '__len__') and len(output) == 2:
        t = output[0]
        x = output[1]
    else:
        t = output
        x = None  # self.test.X[action, :]
    return t, x
