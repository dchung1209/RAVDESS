import json 
import os

json_file_path = os.path.join('.', 'config', '1DCNNLSTM.json')

with open(json_file_path) as f:
    config = json.load(f)
    print(f"{config.model}")

