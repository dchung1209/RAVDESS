import os
import json
import argparse
from model.CNNX import CNNX

class ConfigProcess():
    def __init__(self, name):
        self.name = name
        self.file_path = os.path.join('.', 'config', name + '.json')
        self.epoch = 0
        self.optimizer = ""
        self.loss = ""
    
    def read(self):
        with open(self.file_path) as f:
            config = json.load(f)
            self.epoch = config["epoch"]
            self.optimizer = config["optimizer"]
            self.loss = config["loss"]
    
    def get_model(self):
        return self.name
    
    def get_epoch(self):
        return self.epoch
    
    def get_optimizer(self):
        return self.optimizer['type']

    def get_learning_rate(self):
        return self.optimizer['lr']
        
    def get_loss(self):
        return self.loss




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", action="store")
    args = parser.parse_args()
    instance = ConfigProcess(args.m)
    instance.read()
    print(instance.get_model())
    print(instance.get_epoch())
    print(instance.get_optimizer())
    print(instance.get_learning_rate())


    


