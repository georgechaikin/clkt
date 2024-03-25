"""
This module defines the config class for parameters access from .yaml files.
"""

from os import PathLike
from pathlib import Path
from typing import Union, List, Dict

import yaml

YAMLType = Union[Dict[str, 'YAMLType'], List['YAMLType'], str, int, float, bool, None]


class ConfigFile:
    """Class for config parameters preprocessing.

    Additional parameters can be added while using ConfigFile object.

    """

    def __init__(self, *config_paths: Union[str, PathLike]) -> None:
        """Creates the union config file using the list of config files.

        Args:
            config_paths: config paths that are used for parameters updating.
        """
        self.config = {}
        for config_path in config_paths:
            self.update_config(config_path)

    def update_config(self, config_path: Union[str, PathLike]) -> None:
        """Updates the config file.

        Args:
            config_path: Path to the config file.

        """
        with open(config_path, encoding='utf-8') as f:
            new_data = yaml.safe_load(f)
            shared_keys = self.config.keys() & new_data.keys()
            if shared_keys:
                raise ValueError(f'Shared keys noticed when '
                                 f'updating configuration file: {shared_keys}')
            self.config.update(new_data)

    def __getattr__(self, item) -> YAMLType:
        return self.config.get(item, None)


sys_config_path = Path(__file__) / 'config.yaml'
sys_config = ConfigFile(sys_config_path)
