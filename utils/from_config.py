import importlib
from typing import Callable


def get_module_from_string(module_string: str) -> Callable:
    """returns a module from a string

    Parameters
    ----------
    module_string : str
        The string identifying the module

    Returns
    -------
    Callable
        The module
    """
    module, class_name = module_string.rsplit(".", 1)

    return getattr(importlib.import_module(module), class_name)


def init_from_config(
    object: Callable, config: dict, additional_params: dict = None
) -> Callable:
    """instantiates an object from a config dictionary

    Parameters
    ----------
    object : Callable
        Object to be instantiated
    config : dict
        Configuration dictionary to be used for instantiation
    additional_params : dict, optional
        Additional parameters to use that where not present in the config file,
        by default None

    Returns
    -------
    Callable
        The initialized object
    """
    new_config = {}

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

    object = object(**new_config)

    return object
