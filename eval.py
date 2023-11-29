from sklearn.model_selection import KFold
from preprocess import DataModule
from trainer import Trainer
import torch

class kFold():
    def __init__(self, optimizer, model, loss, k = 4):
        self.actor = [x for x in range(1, 25)]
        self.kf = KFold(n_splits=k, shuffle=True, random_state = 351)
        self.optimizer = optimizer
        self.loss = loss
        self.device = torch.device("cuda")
        self.model = model.to(self.device)
    
    def report(model, dataloader):
        model.eval()
        y_true=[]
        y_pred=[]
        labels = ['Neutral','Calm','Happy','Sad','Angry','Fearful','Disgust','Surprised']

        with torch.inference_mode():
            for b, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            y_logits = model(X)
            y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
            y_true.append(y.cpu())
            y_pred.append(y_preds.cpu())

  true = np.concatenate(y_true)
  pred = np.concatenate(y_pred)

  return classification_report(true, pred, digits=4, output_dict = True)

    def eval(self):
        for i, (train_index, valid_index) in enumerate(self.kf.split(self.actor)):
            train_actor, valid_actor = [self.actor[j] for j in train_index], [self.actor[j] for j in valid_index]

            for epoch in range(200):
                self.trainer.train_step()
                self.trainer.eval_step()
    
