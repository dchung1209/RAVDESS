import os
import json
import argparse
import importlib
import torchvision.transforms as transforms
from trainer import Trainer
from preprocess import DataModule
from sklearn.model_selection import KFold


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
            
            return True

        return False
    
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

    config = ConfigProcess(args.m)
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(128, 256)),
            transforms.ToTensor()])
    module = DataModule(r"c:\Users\dchung\Downloads\archive (1)", 1, transform)
    module.transform_dataframe()
    actor = [x for x in range(1, 25)]

    """
    if (config.read()):
        kf = KFold(n_splits=4, shuffle=True, random_state = 351)
        model_module = importlib.import_module(f"model.{config.get_model()}")
        model_class = getattr(model_module, config.get_model())

        for i, (train_index, valid_index) in enumerate(kf.split(actor)):
            print(f'FOLD {i + 1}')
            print('--------------------------------')
            train_actor, valid_actor = [actor[j] for j in train_index], [actor[j] for j in valid_index]
            train_dataloader, valid_dataloader = module.get_si_dataloader(train_actor, valid_actor)

            instance = Trainer(model_class(), 
                         config.get_optimizer(),
                         config.get_learning_rate(),
                         config.get_loss())
            
            for epoch in range(config.get_epoch()):
                print(f"epoch {epoch + 1}: ")
                instance.train_step(train_dataloader)
                instance.eval_step(valid_dataloader)
                """
    

        
        
        
    

    


