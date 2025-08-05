import yaml
import os

def load_yaml(path):
    path = os.path.join(".\_internal", path)
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        return data
    return None