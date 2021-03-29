import torch
from torch import nn
from Config import Config
import DataProcess
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, pretrained_vec, embedding_size=Config.embeding_size, hidden_size=Config.hidden_size,
                 num_layers = Config.num_layers, drop_prob = Config.drop_prob, n_class = Config.n_class,
                 update_w2v = Config.update_w2v, bidirection = Config.bidirection):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = embedding_size
        self.n_class = n_class
        self.layers = num_layers
        self.bidirection = bidirection

        self.embedding = nn.Embedding.from_pretrained(pretrained_vec)
        self.embedding.weight.requires_grad = update_w2v
        self.encoder = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=bidirection, dropout=drop_prob)

        if self.bidirection:
            self.decoder1 = nn.Linear(self.hidden_size * 4, self.hidden_size)
            self.decoder2 = nn.Linear(self.hidden_size, n_class)
        else:
            self.decoder1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.decoder2 = nn.Linear(self.hidden_size, n_class)

    def forward(self, input):
        embeddings = self.embedding(input)      # [batch_size, seq_len] -> [batch_size, seq_len, embedding_size]
        outputs, state = self.encoder(embeddings.permute(1, 0, 2))
        output = torch.cat([outputs[0], outputs[-1]], dim=1)
        output = self.decoder1(output)
        output = self.decoder2(output)
        return output


class LSTM_attention(nn.Module):
    def __init__(self, pretrained_vec, embedding_size=Config.embeding_size, hidden_size=Config.hidden_size,
                 num_layers=Config.num_layers, drop_prob=Config.drop_prob, n_class=Config.n_class,
                 update_w2v=Config.update_w2v, bidirection=Config.bidirection):
        super(LSTM_attention, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = embedding_size
        self.n_class = n_class
        self.layers = num_layers
        self.bidirection = bidirection

        self.embedding = nn.Embedding.from_pretrained(pretrained_vec)
        self.embedding.weight.requires_grad = update_w2v
        self.encoder = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                               bidirectional=bidirection, dropout=drop_prob, batch_first=True)

        if self.bidirection:
            self.att_weight = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.hidden_size * 2))
            self.att_proj = nn.Parameter(torch.Tensor(self.hidden_size * 2, 1))
            self.decoder1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.decoder2 = nn.Linear(self.hidden_size, n_class)
        else:
            self.att_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
            self.att_proj = nn.Parameter(torch.Tensor(self.hidden_size, 1))
            self.decoder1 = nn.Linear(self.hidden_size, self.hidden_size)
            self.decoder2 = nn.Linear(self.hidden_size, n_class)

        nn.init.uniform_(self.att_weight, -0.1, -0.1)
        nn.init.uniform_(self.att_proj, -0.1, 0.1)

    def forward(self, input):
        embeddings = self.embedding(input)  # [batch_size, seq_len] -> [batch_size, seq_len, embedding_size]
        outputs, state = self.encoder(embeddings)   # [b_s, seq_len, embedding_size]
        attention = torch.tanh(torch.matmul(outputs, self.att_weight))      # [b_s, seq_len, hidden_size(*2)]
        attention = torch.matmul(attention, self.att_proj)      # [b_s, seq_len, 1]
        score = F.softmax(attention, dim=1)     # [b_s, seq_len, 1]
        outputs = outputs * score       # [b_s, seq_len, 100(*2)]
        output = torch.sum(outputs, dim=1)      # [b_s, 100(*2)]

        output = self.decoder1(output)
        output = self.decoder2(output)
        return output




if __name__ == "__main__":
    word2id = DataProcess.build_word2id()
    word2vec = DataProcess.build_word2vec(word2id)
    word2vec = torch.from_numpy(word2vec).float()
    model = LSTM_attention(word2vec)
    print(model)