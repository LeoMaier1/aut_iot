import json
import os

def load_config(path="config.json"):
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(path, "r") as file:
        config = json.load(file)
    return config