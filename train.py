def cal_avg(l):
  Accuracy = 0
  Recall = 0
  F1Score = 0
  Precision = 0
  support = 0
  for report in l:
    support += 1
    Accuracy += report['accuracy']
    Recall += report['macro avg']['recall']
    F1Score += report['macro avg']['f1-score']
    Precision += report['macro avg']['precision']

    print(f"Fold {support}: Accuracy : {round(report['accuracy'], 4)} Precision : {round(report['macro avg']['precision'],4)} Recall : {round(report['macro avg']['recall'], 4)} F1-Score : {round(report['macro avg']['f1-score'],4)}")
  Accuracy /= support
  Recall /= support
  F1Score /= support
  Precision /= support
  return round(Accuracy, 4), round(Recall, 4), round(F1Score, 4), round(Precision, 4)


def accuracy(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(model, dataloader, optim, loss_fn, accuracy_fn):
    train_loss = 0.0
    train_acc = 0.0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        y_logits = model(X)
        y_preds = torch.log_softmax(y_logits, dim=1).argmax(dim=1)

        acc = accuracy_fn(y_preds, y)
        loss = loss_fn(y_logits, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item()
        train_acc += acc

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    return train_loss



def eval_step(model, dataloader, optim, loss_fn, accuracy_fn):
    test_loss = 0.0
    test_acc = 0.0

    model.eval()
    with torch.inference_mode():
        for b, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            y_logits = model(X)
            y_preds = torch.log_softmax(y_logits, dim=1).argmax(dim=1)

            acc = accuracy_fn(y_preds, y)
            loss = loss_fn(y_logits, y)

            test_loss += loss.item()
            test_acc += acc
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

    return test_loss

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
