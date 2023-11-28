from sklearn.model_selection import KFold
from preprocess import DataModule
from trainer import Trainer
import torch

class kFold():
    def __init__(self, optimizer, model, loss, learning_rate, k = 4):
        self.actor = [x for x in range(1, 25)]
        self.kf = KFold(n_splits=k, shuffle=True, random_state = 351)
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.device = torch.device("cuda")
        self.trainer = Trainer(model, )

    def eval(self):
        for i, (train_index, valid_index) in enumerate(self.kf.split(self.actor)):
            train_actor, valid_actor = [self.actor[j] for j in train_index], [self.actor[j] for j in valid_index]

            for epoch in range(200):
                self.trainer.train_step()
                self.trainer.eval_step()
    
    