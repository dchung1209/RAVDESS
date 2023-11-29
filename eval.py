from sklearn.model_selection import KFold
from preprocess import DataModule
from trainer import Trainer
import torch

class kFold(Trainer):
    def __init__(self, model, optimizer, learning_rate, loss, k = 4):
        super().__init__()
        self.actor = [x for x in range(1, 25)]
        self.kf = KFold(n_splits=k, shuffle=True, random_state = 351)
        self.loss = getattr(torch.nn, loss)
        self.model = model
        self.lr = learning_rate
        self.device = torch.device("cuda")

    def get_model(self):
        return self.model
    
    def run(self):
        print(torch.cuda.is_available())
        for i, (train_index, valid_index) in enumerate(self.kf.split(self.actor)):
            print(f'FOLD {i + 1}')
            print('--------------------------------')
            train_actor, valid_actor = [self.actor[j] for j in train_index], [self.actor[j] for j in valid_index]
            self.train_step()

