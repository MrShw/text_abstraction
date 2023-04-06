from gensim.models.word2vec import Word2Vec


def save_vocab_as_txt(filename, word_to_id):

    with open(filename, 'w', encoding='utf-8') as f:
        for k, v in word_to_id.items():
            f.write("{}\t{}\n".format(k, v))

def get_vocab_from_model(vocab_path, reverse_vocab_path):

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


def load_embedding_matrix_from_model(wv_model_path):
    wv_model = Word2Vec.load(wv_model_path)
    embedding_matrix = wv_model.wv.vectors

    return embedding_matrix