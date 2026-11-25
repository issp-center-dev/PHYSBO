# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import warnings

import numpy as np
import configparser


class SetConfig:
    def __init__(self, search_config=None, learning_config=None, config_file=None):
        """
        Setting configuration for search and learning.

        Parameters
        ----------
        search_config: physbo.misc.Search object
        learning_config: physbo.misc.Learning object
        config_file: str, optional
            If given, load values from the configuration file.
        """

        if config_file is not None:
            if search_config is not None or learning_config is not None:
                raise ValueError("search_config and learning_config must be None if config_file is given.")
            self.load(config_file)
            return

        if search_config is None:
            search_config = Search()
        self.search = search_config

        if learning_config is None:
            learning_config = Adam()
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
        config = configparser.ConfigParser()
        loaded_files = config.read(file_name)
        if not loaded_files:
            raise FileNotFoundError(f"Configuration file is not found: {file_name}")

        learning_section = _get_section(config, "learning")
        if learning_section is not None:
            method = learning_section.get("method", "adam").strip().lower()
        else:
            method = "adam"

        self.search = Search(config)

        if method == "adam":
            self.learning = Adam(config)
        elif method in ("bfgs", "batch"):
            self.learning = Batch(config)
        else:
            raise ValueError(
                f"Unknown learning method '{method}'. Supported methods are: adam, bfgs, batch."
            )


class Search:
    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : configparser.ConfigParser, optional
            If given, load values from the [search] section.
        """
        self.multi_probe_num_sampling = 20
        self.alpha = 1.0
        if config is not None:
            self.load(config)

    def load(self, config):
        """
        Loading information of configuration from [search] section.

        Parameters
        ----------
        config : configparser.ConfigParser

        Returns
        -------

        """
        temp_dict = _get_section(config, "search")
        if temp_dict is None:
            return
        self.multi_probe_num_sampling = int(
            temp_dict.get("multi_probe_num_sampling", self.multi_probe_num_sampling)
        )
        self.alpha = np.float64(temp_dict.get("alpha", self.alpha))

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


class Learning(object):
    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : configparser.ConfigParser, optional
            If given, load values from the [learning] section.
        """
        self.is_disp = True
        self.num_disp = 10
        self.num_init_params_search = 20
        if not hasattr(self, "method"):
            self.method = "adam"
        if config is not None:
            self.load(config)

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
        Loading information of configuration from [learning] section.

        Parameters
        ----------
        config : configparser.ConfigParser


        Returns
        -------

        """
        temp_dict = _get_section(config, "learning")
        if temp_dict is None:
            return
        self.method = temp_dict.get("method", self.method).strip().lower()
        self.is_disp = boolean(temp_dict.get("is_disp", self.is_disp))
        self.num_disp = int(temp_dict.get("num_disp", self.num_disp))
        self.num_init_params_search = int(
            temp_dict.get("num_init_params_search", self.num_init_params_search)
        )


class Batch(Learning):
    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : configparser.ConfigParser, optional
            If given, load values from [learning] and [learning.batch] sections.
        """
        self.method = "bfgs"
        self.max_iter = 200
        self.max_iter_init_params_search = 20
        self.batch_size = 5000
        super(Batch, self).__init__(config)

    def show(self):
        """
        Showing information about configuration about batch object.

        Returns
        -------

        """
        super(Batch, self).show()
        print("max_iter: ", self.max_iter)
        print("max_iter_init_params_search: ", self.max_iter_init_params_search)
        print("batch_size: ", self.batch_size)

    def load(self, config):
        """
        Loading information of configuration from [learning.batch] section.

        Parameters
        ----------
        config : configparser.ConfigParser

        Returns
        -------

        """
        super(Batch, self).load(config)
        temp_dict = _get_learning_section(config, "batch")
        if temp_dict is None:
            return
        self.max_iter = int(temp_dict.get("max_iter", self.max_iter))
        self.max_iter_init_params_search = int(
            temp_dict.get("max_iter_init_params_search", self.max_iter_init_params_search)
        )
        self.batch_size = int(temp_dict.get("batch_size", self.batch_size))


class Online(Learning):
    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : configparser.ConfigParser, optional
            If given, load values from [learning] and [learning.online] sections.
        """
        self.max_epoch = 500
        self.max_epoch_init_params_search = 50
        self.batch_size = 64
        self.eval_size = 5000
        super(Online, self).__init__(config)

    def show(self):
        """
        Showing information about configuration about online object.

        Returns
        -------

        """
        super(Online, self).show()
        print("max_epoch: ", self.max_epoch)
        print("max_epoch_init_params_search: ", self.max_epoch_init_params_search)
        print("batch_size: ", self.batch_size)
        print("eval_size: ", self.eval_size)

    def load(self, config):
        """
        Loading information of configuration from [learning.online] section.

        Parameters
        ----------
        config : configparser.ConfigParser


        Returns
        -------

        """
        super(Online, self).load(config)
        temp_dict = _get_learning_section(config, "online")
        if temp_dict is None:
            return
        self.max_epoch = int(temp_dict.get("max_epoch", self.max_epoch))
        self.max_epoch_init_params_search = int(
            temp_dict.get("max_epoch_init_params_search", self.max_epoch_init_params_search)
        )
        self.batch_size = int(temp_dict.get("batch_size", self.batch_size))
        self.eval_size = int(temp_dict.get("eval_size", self.eval_size))


class Adam(Online):
    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : configparser.ConfigParser, optional
            If given, load values from [learning], [learning.online], and [learning.adam] sections.
        """
        self.method = "adam"
        self.alpha = 0.001
        self.beta = 0.9
        self.gamma = 0.999
        self.epsilon = 1e-6
        super(Adam, self).__init__(config)

    def show(self):
        """
        Showing information about configuration about adam object.

        Returns
        -------

        """
        super(Adam, self).show()
        print("alpha = ", self.alpha)
        print("beta = ", self.beta)
        print("gamma = ", self.gamma)
        print("epsilon = ", self.epsilon)
        print("\n")

    def load(self, config):
        """
        Loading information of configuration from [learning.adam] section.

        Parameters
        ----------
        config : configparser.ConfigParser

        Returns
        -------

        """
        super(Adam, self).load(config)
        temp_dict = _get_learning_section(config, "adam")
        if temp_dict is None:
            return
        self.alpha = np.float64(temp_dict.get("alpha", self.alpha))
        self.beta = np.float64(temp_dict.get("beta", self.beta))
        self.gamma = np.float64(temp_dict.get("gamma", self.gamma))
        self.epsilon = np.float64(temp_dict.get("epsilon", self.epsilon))


def _get_section(config, section_name):
    """Return section dict if present, else None. Missing sections are skipped (defaults used)."""
    if not config.has_section(section_name):
        return None
    return config[section_name]


def _get_learning_section(config, name):
    """
    Get a learning sub-section by name (batch, online, or adam).
    Prefer [learning.name]; fall back to [name] with deprecation warning.
    Return None if neither section exists (defaults will be used).
    """
    new_name = f"learning.{name}"
    old_name = name
    if config.has_section(new_name):
        return config[new_name]
    if config.has_section(old_name):
        warnings.warn(
            f"Section [{old_name}] is deprecated and will be removed in a future version. "
            f"Use [{new_name}] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return config[old_name]
    return None


def boolean(value):
    """
    Return boolean.

    Parameters
    ----------
    value: str or boolean

    Returns
    -------
    True or False
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False

    if value in (0, 1):
        return bool(value)

    raise ValueError(f"Cannot parse boolean value: {value!r}")
