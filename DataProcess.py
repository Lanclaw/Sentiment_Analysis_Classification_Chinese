from Config import Config
import re
import gensim
import os
import numpy as np
from torch.utils.data import Dataset
import torch


class Data_set(Dataset):
    def __init__(self, data, label):
        self.data = data
        if label is not None:
            self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.from_numpy(self.data[index])
        if self.label is not None:
            label = torch.from_numpy(self.label[index])
            return data, label
        return data


def stopword_list():
    f = open(Config.stopword_path, encoding='UTF-8').readlines()
    stoplist = [line.strip() for line in f]
    return stoplist


def build_word2id(file=Config.word2id_path):
    if os.path.exists(Config.word2id_path):
        word2id = {}
        word2id_list = []
        with open(Config.word2id_path, encoding='UTF-8') as f:
            for line in f:
                line = line.strip().split()
                word2id_list.append(line)
            word2id = dict(word2id_list)

        for key in word2id:
            word2id[key] = int(word2id[key])
    else:
        word2id = {'_PAD_': 0}
        stoplist = stopword_list()
        paths = [Config.train_path, Config.val_path]
        for path in paths:
            f = open(path, encoding='UTF-8').readlines()
            for line in f:
                line = line.strip().split()[1:]
                for word in line:
                    if word not in word2id.keys() and word not in stoplist and len(re.findall('[a-zA-Z]+', word)) == 0:
                        word2id[word] = len(word2id)

        with open(file, 'w', encoding='UTF-8') as f:
            for w in word2id:
                f.write(w + '\t')
                f.write(str(word2id[w]))
                f.write('\n')
    return word2id


def build_word2vec(word2id, pre_vec=Config.pre_word2vec_path):
    word2vec = np.random.uniform(-1, 1, [len(word2id), Config.embeding_size])
    if os.path.exists(Config.word2vec_path):
        with open(Config.word2vec_path, encoding='UTF-8') as f:
            for i, vec in enumerate(f):
                vec = vec.strip().split()
                for j, feat in enumerate(vec):
                    vec[j] = float(feat)
                word2vec[i] = vec
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(pre_vec, binary=True)
        for word in word2id.keys():
            try:
                word2vec[word2id[word]] = model[word]
            except KeyError:
                pass
        with open(Config.word2vec_path, 'w', encoding='UTF-8') as f:
            for vec in word2vec:
                vec = [str(feature) for feature in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word2vec


def text2array(word2id, path, path_data, path_label, max_seq_len=Config.max_seq_len):

    if os.path.exists(path_data):
        text2array = np.loadtxt(path_data)
        labels = np.loadtxt(path_label)
        labels = np.array([labels]).T
    else:
        text2array = []

        with open(path, encoding='UTF-8') as f:
            sum_seq = len(f.readlines())

        text2array = np.zeros(shape=(sum_seq, max_seq_len))
        labels = []
        with open(path, encoding='UTF-8') as f:
            for i, line in enumerate(f.readlines()):
                label = line.strip().split()[0]
                line = line.strip().split()[1:]
                new_sen = [word2id.get(word, 0) for word in line]
                if len(new_sen) >= max_seq_len:
                    text2array[i] = new_sen[0:max_seq_len]
                else:
                    text2array[i, max_seq_len - len(new_sen):] = new_sen

                labels.append(int(label))


        np.savetxt(path_data, text2array, fmt='%d')
        np.savetxt(path_label, labels, fmt='%d')
        labels = np.array([labels]).T

    return text2array, labels


def text2array_nolabel(word2id, path, max_seq_len=Config.max_seq_len):

    with open(path, encoding='UTF-8') as f:
        sum_seq = len(f.readlines())

    text2array = np.zeros(shape=(sum_seq, max_seq_len))
    with open(path, encoding='UTF-8') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split()
            new_sen = [word2id.get(word, 0) for word in line]
            if len(new_sen) >= max_seq_len:
                text2array[i] = new_sen[0:max_seq_len]
            else:
                text2array[i, max_seq_len - len(new_sen):] = new_sen

    return text2array


def prepare_data(word2id):
    train_data, train_label = text2array(word2id, Config.train_path, Config.train_data_path, Config.train_label_path)
    val_data, val_label = text2array(word2id, Config.val_path, Config.val_data_path, Config.val_label_path)
    test_data, test_label = text2array(word2id, Config.test_path, Config.test_data_path, Config.test_label_path)
    return train_data, train_label, val_data, val_label, test_data, test_label




