"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
from typing import Dict

from omegaconf import OmegaConf


class Config:
    def __init__(self, args):
        self.config = {}

        self.args = args

        user_config = self._build_opt_list(self.args.options)

        config = OmegaConf.load(self.args.cfg_path)

        runner_config = self.build_runner_config(config)

        # Validate the user-provided runner configuration
        # model and dataset configuration are supposed to be validated by the respective classes
        # [TODO] validate the model/dataset configuration
        # self._validate_runner_config(runner_config)

        # Override the default configuration with user options.
        self.config = OmegaConf.merge(
            runner_config, user_config
        )

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)


    @staticmethod
    def build_runner_config(config):
        return {"run": config.run}


    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def get_config(self):
        return self.config

    @property
    def run_cfg(self):
        return self.config.run

    @property
    def datasets_cfg(self):
        return self.config.datasets

    @property
    def model_cfg(self):
        return self.config.model

    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))

        logging.info("\n======  Dataset Attributes  ======")
        datasets = self.config.datasets

        for dataset in datasets:
            if dataset in self.config.datasets:
                logging.info(f"\n======== {dataset} =======")
                dataset_config = self.config.datasets[dataset]
                logging.info(self._convert_node_to_json(dataset_config))
            else:
                logging.warning(f"No dataset named '{dataset}' in config. Skipping")

        logging.info(f"\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.model))

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)


def node_to_dict(node):
    return OmegaConf.to_container(node)


class ConfigValidator:
    """
    This is a preliminary implementation to centralize and validate the configuration.
    May be altered in the future.

    A helper class to validate configurations from yaml file.

    This serves the following purposes:
        1. Ensure all the options in the yaml are defined, raise error if not.
        2. when type mismatches are found, the validator will raise an error.
        3. a central place to store and display helpful messages for supported configurations.

    """

    class _Argument:
        def __init__(self, name, choices=None, type=None, help=None):
            self.name = name
            self.val = None
            self.choices = choices
            self.type = type
            self.help = help

        def __str__(self):
            s = f"{self.name}={self.val}"
            if self.type is not None:
                s += f", ({self.type})"
            if self.choices is not None:
                s += f", choices: {self.choices}"
            if self.help is not None:
                s += f", ({self.help})"
            return s

    def __init__(self, description):
        self.description = description

        self.arguments = dict()

        self.parsed_args = None

    def __getitem__(self, key):
        assert self.parsed_args is not None, "No arguments parsed yet."

        return self.parsed_args[key]

    def __str__(self) -> str:
        return self.format_help()

    def add_argument(self, *args, **kwargs):
        """
        Assume the first argument is the name of the argument.
        """
        self.arguments[args[0]] = self._Argument(*args, **kwargs)

    def validate(self, config=None):
        """
        Convert yaml config (dict-like) to list, required by argparse.
        """
        for k, v in config.items():
            assert (
                k in self.arguments
            ), f"""{k} is not a valid argument. Support arguments are {self.format_arguments()}."""

            if self.arguments[k].type is not None:
                try:
                    self.arguments[k].val = self.arguments[k].type(v)
                except ValueError:
                    raise ValueError(f"{k} is not a valid {self.arguments[k].type}.")

            if self.arguments[k].choices is not None:
                assert (
                    v in self.arguments[k].choices
                ), f"""{k} must be one of {self.arguments[k].choices}."""

        return config

    def format_arguments(self):
        return str([f"{k}" for k in sorted(self.arguments.keys())])

    def format_help(self):
        # description + key-value pair string for each argument
        help_msg = str(self.description)
        return help_msg + ", available arguments: " + self.format_arguments()

    def print_help(self):
        # display help message
        print(self.format_help())
