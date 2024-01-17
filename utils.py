import json
import typing
def read_json_config(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON in '{config_file}'.")
        return {}


def initialize_with_config(class_obj:typing.Callable, config_file:str,type:str=None,**kwargs):
    config = read_json_config(config_file)
    if type is not None:
        config=config[type]
    config.update(kwargs)
    return class_obj(**config)



