import DataProcess
import torch
from torch.utils.data import DataLoader
from Config import Config
from sklearn.metrics import f1_score, recall_score
import model
from torch import nn


def val_acc(val_loader, model, device):
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    val_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            data = data.type(torch.LongTensor)
            data = data.to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)
            label = label.squeeze()
            output = model(data)
            loss = loss_func(output, label)

            _, pred = torch.max(output, dim=1)
            val_loss += loss
            total += len(label)
            correct += (pred == label).sum().item()
            f1 = f1_score(label.cpu(), pred.cpu(), average='weighted')
            recall = recall_score(label.cpu(), pred.cpu(), average='micro')
        print('val_loss: {:.3f}, val_acc: {:.3f}, f1_score: {:.3f}, recall_score: {:.3f}'.format(
            val_loss / (i + 1), 100 * correct / total, f1, recall))
        return correct / total


def test_acc(test_loader, model, device):
    pass


def predict(word2id, model, path=Config.pre_path):
    model.cpu()
    ret = ['pos', 'neg']
    with torch.no_grad():
        input = torch.from_numpy(DataProcess.text2array_nolabel(word2id, path))
        input = input.type(torch.LongTensor)
        output = model(input)
        _, pred = torch.max(output, dim=1)
        with open(path, encoding='UTF-8') as f:
            for i, line in enumerate(f.readlines()):
                print(ret[pred[i].item()], line)




