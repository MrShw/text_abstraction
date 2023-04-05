# step4
# gensim是一个Python库，用于实现文本处理和自然语言处理的各种算法和模型，其中包括Word2Vec词向量模型。
from gensim.models.word2vec import Word2Vec


def save_vocab_as_txt(filename, word_to_id):
    """
    :param filename:txt文件路径
    :param word_to_id:要保存的字典
    :return: None
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for k, v in word_to_id.items():
            f.write("{}\t{}\n".format(k, v))


# 从word2vec模型中获取正向和反向词典
def get_vocab_from_model(vocab_path, reverse_vocab_path):
    # 提取映射字典
    # vocab_path: word_to_id的文件存储路径
    # reverse_vocab_path: id_to_word的文件存储路径
    word_to_id, id_to_word = {}, {}
    with open(vocab_path, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            w, v = line.strip('\n').split('\t')
            word_to_id[w] = int(v)

    with open(reverse_vocab_path, 'r', encoding='utf-8') as f2:
        for line in f2.readlines():
            v, w = line.strip('\n').split('\t')
            id_to_word[int(v)] = w

    return word_to_id, id_to_word


# 从word2vec模型中获取词向量矩阵
def load_embedding_matrix_from_model(wv_model_path):
    # 从word2vec模型中获取词向量矩阵
    # wv_model_path: word2vec模型的路径
    wv_model = Word2Vec.load(wv_model_path)
    # wv_model.wv.vectors包含词向量矩阵
    embedding_matrix = wv_model.wv.vectors
    return embedding_matrix