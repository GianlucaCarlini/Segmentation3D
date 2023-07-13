import importlib
from typing import Callable, Union
import warnings


def get_module_from_string(module_string: str) -> Callable:
    """returns a module from a string

    Parameters
    ----------
    module_string : str
        The string identifying the module

    Returns
    -------
    module : Callable
        The module
    """
    module, class_name = module_string.rsplit(".", 1)

    return getattr(importlib.import_module(module), class_name)


def init_from_config(
    config: dict,
    object: Callable = None,
    additional_params: dict = None,
    return_config_only: bool = False,
) -> Union[Callable, dict]:
    """instantiates an object from a config dictionary

    Parameters
    ----------
    config : dict
        Configuration dictionary to be used for instantiation
    object : Callable, optional
        Object to be instantiated
    additional_params : dict, optional
        Additional parameters to use that where not present in the config file,
        by default None
    return_config_only : bool, optional
        Wheter if return the config dictionary or the instantiated object, by default False

    Returns
    -------
    object : Callable
        The initialized object, only if return_config_only is False
    config : dict
        The configuration dictionary, only if return_config_only is True
    """
    new_config = {}

    if object is None and return_config_only is False:
        warnings.warn(
            "No object was provided and return_config_only is False, setting return_config_only to True"
        )
        return_config_only = True

    for key, value in config.items():
        if value.get("target") is not None:
            target = value.get("target")
            if not isinstance(target, str):
                temp = {}
                for k, v in target.items():
                    temp[k] = get_module_from_string(v)(**value.get("params")[k])
                new_config[key] = temp
                continue
            module = get_module_from_string(target)
            if value.get("params") is not None:
                params = value.get("params")
                new_config[key] = module(**params)
            else:
                new_config[key] = module
        else:
            new_config.update(dict(value))

    if additional_params is not None:
        new_config.update(additional_params)

    if return_config_only:
        return new_config

    object = object(**new_config)

    return object
