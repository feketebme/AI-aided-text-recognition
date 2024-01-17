from utils import read_json_config
import pytorch_lightning as pl

# List of callback names
callback_names = [
    "model_checkpoint",
    "early_stopping",
    "your_custom_callback",
    "another_custom_callback",
    "yet_another_callback",
    "and_so_on_callback"
]


def get_callbacks(callback_config_path):
    callbacks_config = read_json_config(callback_config_path)
    callbacks = []
    # Iterate over callback names and add them if present in the JSON file
    for callback_name in callback_names:
        if callback_name in callbacks_config:
            callback_instance = getattr(pl.callbacks, callback_name.capitalize())(**callbacks_config[callback_name])
            callbacks.append(callback_instance)
    return callbacks
