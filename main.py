import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import DataProcess
import model
from Config import Config
from sklearn.metrics import f1_score, recall_score
import eval


def train(train_dataloader, model, device, epoches=Config.n_epoches, lr=Config.lr):
    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epoches):
        train_loss = 0
        correct = 0
        total = 0

        train_dataloader = tqdm.tqdm(train_dataloader)
        for i, (data, label) in enumerate(train_dataloader):
            data = data.type(torch.LongTensor)
            data = data.to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)
            label = label.squeeze()
            output = model(data)
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(output, dim=1)
            train_loss += loss
            total += len(label)
            correct += (pred == label).sum().item()
            f1 = f1_score(label.cpu(), pred.cpu(), average='weighted')
            recall = recall_score(label.cpu(), pred.cpu(), average='micro')
            postfix = {'train_loss: {:.3f}, train_acc: {:.3f}, f1_score: {:.3f}, recall_score: {:.3f}'.format(
                train_loss / (i + 1), 100 * correct / total, f1, recall
            )}
            train_dataloader.set_postfix(log=postfix)


if __name__ == '__main__':
    word2id = DataProcess.build_word2id()
    word2vec = torch.from_numpy(DataProcess.build_word2vec(word2id)).float()
    model = model.LSTM(word2vec)
    print(model)
    train_data, train_label, val_data, val_label, test_data, test_label = DataProcess.prepare_data(word2id)
    train_loader = DataProcess.Data_set(train_data, train_label)
    train_loader = DataLoader(train_loader, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataProcess.Data_set(val_data, val_label)
    val_loader = DataLoader(val_loader, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataProcess.Data_set(test_data, test_label)
    test_loader = DataLoader(test_loader, batch_size=Config.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(train_loader, model, device, epoches=1)
    eval.val_acc(val_loader, model, device)

