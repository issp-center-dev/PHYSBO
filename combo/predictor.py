import numpy as np
import cPickle as pickle
import gp

class base_predictor( object ):
    """

    """
    def __init__( self, config, model = None ):
        """

        Parameters
        ----------
        config
        model
        """

        self.config = config
        self.model = model
        if self.model is None:
            self.model = gp.core.model(cov = gp.cov.gauss( num_dim = None, ard = False ), mean = gp.mean.const(), lik = gp.lik.gauss())

    def fit( self, *args, **kwds ):
        """

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

        Parameters
        ----------
        file_name

        Returns
        -------

        """
        with open(file_name, 'w') as f:
            pickle.dump(self.__dict__, f, 2)

    def load(self, file_name):
        """

        Parameters
        ----------
        file_name

        Returns
        -------

        """
        with open(file_name) as f:
            tmp_dict = pickle.load(f)
            self.update(tmp_dict)
