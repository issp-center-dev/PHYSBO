# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import configparser


class set_config:
    def __init__(self, search_config=None, learning_config=None):
        """
        Setting configuration for search and learning.

        Parameters
        ----------
        search_config: physbo.misc.search object
        learning_config: physbo.misc.learning object

        """
        if search_config is None:
            search_config = search()
        self.search = search_config

        if learning_config is None:
            learning_config = adam()
        self.learning = learning_config

    def show(self):
        """
        Showing information of search and learning objects.

        Returns
        -------

        """
        self.search.show()
        self.learning.show()

    def load(self, file_name="config.ini"):
        """
        Loading information of configuration.

        Parameters
        ----------
        file_name: str
        An input file name of configuration.
        Returns
        -------

        """
        config = configparser.SafeConfigParser()
        config.read(file_name)

        search_config = search()
        self.search = search_config
        self.search.load(config)

        temp_dict = config._sections["learning"]
        method = temp_dict.get("method", "adam")

        if method == "adam":
            learning_config = adam()
            self.learning = learning_config
            self.learning.load(config)

        if method in ("bfgs", "batch"):
            learning_config = batch()
            self.learning = learning_config
            self.learning.load(config)


class search:
    def __init__(self):
        self.multi_probe_num_sampling = 20
        self.alpha = 1.0

    def load(self, config):
        """
        Loading information of configuration from config._sectoins['search'].

        Parameters
        ----------
        config: physbo.misc.set_config object

        Returns
        -------

        """
        temp_dict = config._sections["search"]
        self.multi_probe_num_sampling = int(
            temp_dict.get("multi_probe_num_sampling", 20)
        )
        self.alpha = np.float64(temp_dict.get("alpha", 1.0))

    def show(self):
        """
        Showing information about search object.

        Returns
        -------

        """
        print("(search)")
        print("multi_probe_num_sampling: ", self.multi_probe_num_sampling)
        print("alpha: ", self.alpha)
        print("\n")


class learning(object):
    def __init__(self):
        self.is_disp = True
        self.num_disp = 10
        self.num_init_params_search = 20
        self.method = "adam"

    def show(self):
        """
        Showing information about learning object.

        Returns
        -------

        """
        print("( learning )")
        print("method : ", self.method)
        print("is_disp: ", self.is_disp)
        print("num_disp: ", self.num_disp)
        print("num_init_params_search: ", self.num_init_params_search)

    def load(self, config):
        """
        Loading information of configuration from config._sectoins['learning'].

        Parameters
        ----------
        config: physbo.misc.set_config object


        Returns
        -------

        """
        temp_dict = config._sections["learning"]
        self.method = temp_dict.get("method", "adam")
        self.is_disp = boolean(temp_dict.get("is_disp", True))
        self.num_disp = int(temp_dict.get("num_disp", 10))
        self.num_init_params_search = int(temp_dict.get("num_init_params_search", 20))


class batch(learning):
    def __init__(self):
        super(batch, self).__init__()
        self.method = "bfgs"
        self.max_iter = 200
        self.max_iter_init_params_search = 20
        self.batch_size = 5000

    def show(self):
        """
        Showing information about configuration about batch object.

        Returns
        -------

        """
        super(batch, self).show()
        print("max_iter: ", self.max_iter)
        print("max_iter_init_params_search: ", self.max_iter_init_params_search)
        print("batch_size: ", self.batch_size)

    def load(self, config):
        """
        Loading information of configuration from config._sectoins['batch'].

        Parameters
        ----------
        config: physbo.misc.set_config object

        Returns
        -------

        """
        super(batch, self).load(config)
        temp_dict = config._sections["batch"]
        self.max_iter = int(temp_dict.get("max_iter", 200))
        self.max_iter_init_params_search = int(
            temp_dict.get("max_iter_init_params_search", 20)
        )
        self.batch_size = int(temp_dict.get("batch_size", 5000))


class online(learning):
    def __init__(self):
        super(online, self).__init__()
        self.max_epoch = 500
        self.max_epoch_init_params_search = 50
        self.batch_size = 64
        self.eval_size = 5000

    def show(self):
        """
        Showing information about configuration about online object.

        Returns
        -------

        """
        super(online, self).show()
        print("max_epoch: ", self.max_epoch)
        print("max_epoch_init_params_search: ", self.max_epoch_init_params_search)
        print("batch_size: ", self.batch_size)
        print("eval_size: ", self.eval_size)

    def load(self, config):
        """
        Loading information of configuration from config._sectoins['online'].

        Parameters
        ----------
        config: physbo.misc.set_config object


        Returns
        -------

        """
        super(online, self).load(config)
        temp_dict = config._sections["online"]
        self.max_epoch = int(temp_dict.get("max_epoch", 1000))
        self.max_epoch_init_params_search = int(
            temp_dict.get("max_epoch_init_params_search", 50)
        )
        self.batch_size = int(temp_dict.get("batch_size", 64))
        self.eval_size = int(temp_dict.get("eval_size", 5000))


class adam(online):
    def __init__(self):
        super(adam, self).__init__()
        self.method = "adam"
        self.alpha = 0.001
        self.beta = 0.9
        self.gamma = 0.999
        self.epsilon = 1e-6

    def show(self):
        """
        Showing information about configuration about adam object.

        Returns
        -------

        """
        super(adam, self).show()
        print("alpha = ", self.alpha)
        print("beta = ", self.beta)
        print("gamma = ", self.gamma)
        print("epsilon = ", self.epsilon)
        print("\n")

    def load(self, config):
        """
        Loading information of configuration from config._sectoins['adam'].

        Parameters
        ----------
        config: physbo.misc.set_config object

        Returns
        -------

        """
        super(adam, self).load(config)
        temp_dict = config._sections["adam"]
        self.alpha = np.float64(temp_dict.get("alpha", 0.001))
        self.beta = np.float64(temp_dict.get("beta", 0.9))
        self.gamma = np.float64(temp_dict.get("gamma", 0.9999))
        self.epsilon = np.float64(temp_dict.get("epsilon", 1e-6))


def boolean(str):
    """
    Return boolean.

    Parameters
    ----------
    str: str or boolean

    Returns
    -------
    True or False
    """
    if str == "True" or str is True:
        return True
    else:
        return False
