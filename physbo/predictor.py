import numpy as np
import cPickle as pickle
import gp

class base_predictor( object ):
    """
        Base predictor is defined in this class.

    """
    def __init__( self, config, model = None ):
        """

        Parameters
        ----------
        config: set_config object (physbo.misc.set_config)
        model: model object
            A default model is set as gp.core.model
        """

        self.config = config
        self.model = model
        if self.model is None:
            self.model = gp.core.model(cov = gp.cov.gauss( num_dim = None, ard = False ), mean = gp.mean.const(), lik = gp.lik.gauss())

    def fit( self, *args, **kwds ):
        """

        Default fit function.
        This function must be overwritten in each model.

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        raise NotImplementedError

    def prepare( self, *args, **kwds ):
        """

        Default prepare function.
        This function must be overwritten in each model.

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        raise NotImplementedError

    def delete_stats( self, *args, **kwds ):
        """

        Default function to delete status
        This function must be overwritten in each model.

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        raise NotImplementedError

    def get_basis( self, *args, **kwds ):
        """

        Default function to get basis
        This function must be overwritten in each model.

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        raise NotImplementedError

    def get_post_fmean( self, *args, **kwds ):
        """

        Default function to get a mean value of the score.
        This function must be overwritten in each model.

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        raise NotImplementedError

    def get_post_fcov( self, *args, **kwds ):
        """

        Default function to get a covariance of the score.
        This function must be overwritten in each model.

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        raise NotImplementedError

    def get_post_params( self,*args, **kwds ):
        """

        Default function to get parameters.
        This function must be overwritten in each model.

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        raise NotImplementedError

    def get_post_samples( self, *args, **kwds ):
        """

        Default function to get samples.
        This function must be overwritten in each model.

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        raise NotImplementedError

    def get_predict_samples( self, *args, **kwds ):
        """

        Default function to get prediction variables of samples.
        This function must be overwritten in each model.

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        raise NotImplementedError

    def get_post_params_samples( self, *args, **kwds ):
        """

        Default function to get parameters of samples.
        This function must be overwritten in each model.

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        raise NotImplementedError

    def update( self,*args, **kwds ):
        """

        Default function to update variables.
        This function must be overwritten in each model.

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        raise NotImplementedError

    def save(self, file_name):
        """

        Default function to save information by using pickle.dump function.
        The protocol version is set as 2.

        Parameters
        ----------
        file_name: str
            A file name to save self.__dict__ object.

        Returns
        -------

        """
        with open(file_name, 'w') as f:
            pickle.dump(self.__dict__, f, 2)

    def load(self, file_name):
        """

        Default function to load variables.
        The information is updated using self.update function.

        Parameters
        ----------
        file_name: str
            A file name to load variables from the file.

        Returns
        -------

        """
        with open(file_name) as f:
            tmp_dict = pickle.load(f)
            self.update(tmp_dict)
