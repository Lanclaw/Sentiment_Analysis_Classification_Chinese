class Config():
    train_path = './data/train.txt'
    val_path = './data/validation.txt'
    test_path = './data/test.txt'
    pre_path = './data/pre.txt'
    word2id_path = './data/word2id.txt'
    word2vec_path = './data/word2vec.txt'
    pre_word2vec_path = './data/wiki_word2vec_50.bin'
    corpus_word2vec_path = './data/word_vec.txt'
    model_state_dict_path = './data/model_state_dict.pkl'   # 训练模型保存的地址
    stopword_path = './data/stopword.txt'
    embeding_size = 50
    max_seq_len = 65
    train_data_path = './data/train_data.txt'
    val_data_path = './data/validation_data.txt'
    test_data_path = './data/test_data.txt'
    train_label_path = './data/train_label.txt'
    val_label_path = './data/validation_label.txt'
    test_label_path = './data/test_label.txt'
    bidirection = True
    update_w2v = True
    num_layers = 2
    hidden_size = 100
    drop_prob = 0.2
    n_class = 2
    lr = 0.0001
    batch_size = 64
    n_epoches = 32
    model_path = './data/model.pkl'


