import torch

class Trainer:
    def __init__(self, model, dataloader, optimizer, loss, accuracy):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.device = torch.device("cuda")

    def accuracy(self, y_pred, y_true):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    def train_step(self):
        train_loss = 0.0
        train_acc = 0.0

        self.model.train()
        for batch, (X, y) in enumerate(self.dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            y_logits = self.model(X)
            y_preds = torch.log_softmax(y_logits, dim=1).argmax(dim=1)

            acc = self.accuracy(y_preds, y)
            loss = self.loss(y_logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_acc += acc

        train_loss /= len(self.dataloader)
        train_acc /= len(self.dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

        return train_loss
    
    def eval_step(self):
        test_loss = 0.0
        test_acc = 0.0

        self.model.eval()
        with torch.inference_mode():
            for b, (X, y) in enumerate(self.dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                y_logits = self.model(X)
                y_preds = torch.log_softmax(y_logits, dim=1).argmax(dim=1)

                acc = self.accuracy(y_preds, y)
                loss = self.loss(y_logits, y)

                test_loss += loss.item()
                test_acc += acc
        test_loss /= len(self.dataloader)
        test_acc /= len(self.dataloader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

        return test_loss
    
