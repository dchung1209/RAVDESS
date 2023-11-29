import torch

class Trainer():
    def __init__(self, model, optimizer, learning_rate, loss):
        self.device = torch.device("cuda")
        self.optimizer = getattr(torch.optim, optimizer)(model.parameters(), learning_rate)
        self.loss = getattr(torch.nn, loss)()
        self.model = model.to(self.device)
        self.lr = learning_rate
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

    def accuracy(self, y_pred, y_true):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    def train_step(self, dataloader):
        train_loss = 0.0
        train_acc = 0.0

        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
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

        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

        self.train_acc.append(train_loss)
        self.train_loss.append(train_acc)
    
    def eval_step(self, dataloader):
        test_loss = 0.0
        test_acc = 0.0

        self.model.eval()
        with torch.inference_mode():
            for b, (X, y) in enumerate(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                y_logits = self.model(X)
                y_preds = torch.log_softmax(y_logits, dim=1).argmax(dim=1)

                acc = self.accuracy(y_preds, y)
                loss = self.loss(y_logits, y)

                test_loss += loss.item()
                test_acc += acc

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

        self.test_acc.append(test_loss)
        self.test_loss.append(test_acc)
    
