import json

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return  f"({self.x},{self.y})"


def do_overlap(l1, r1, l2, r2):
    # If one rectangle is on left side of other
    if l1.x >= r2.x or l2.x >= r1.x:
        return False

    # If one rectangle is above other
    if l1.y <= r2.y or l2.y <= r1.y:
        return False

    return True
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


def initialize_with_config(class_obj, config_file):
    config = read_json_config(config_file)
    return class_obj(**config)
