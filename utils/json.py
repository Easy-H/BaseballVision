import json

def LoadJson(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        return data
    return None