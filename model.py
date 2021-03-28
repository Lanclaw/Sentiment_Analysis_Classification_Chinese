import torch
from torch import nn
from Config import Config
import DataProcess


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


if __name__ == "__main__":
    word2id = DataProcess.build_word2id(Config.word2id_path)
    word2vec = DataProcess.build_word2vec(Config.pre_word2vec_path, word2id)
    word2vec = torch.from_numpy(word2vec)
    model = LSTM(word2vec)
    print(model)