import sys
import os
from collections import Counter
import torch
from torch.utils.data import Dataset

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from pgn_coverage.utils.func_utils import simple_tokenizer, count_words, sort_batch_by_len, source2ids, abstract2ids
from pgn_coverage.utils.vocab import Vocab
from pgn_coverage.utils import config


class PairDataset(object):
    def __init__(self, filename, tokenize=simple_tokenizer, max_enc_len=None, max_dec_len=None,
                 truncate_enc=False, truncate_dec=False):
        print("Reading dataset %s..." % filename, end=' ', flush=True)
        self.filename = filename
        self.pairs = []

        with open(filename, 'r', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                pair = line.strip().split('<SEP>')
                if len(pair) != 2:
                    print("Line %d of %s is error formed." % (i, filename))
                    print(line)
                    continue
                enc = tokenize(pair[0])
                if max_enc_len and len(enc) > max_enc_len:
                    if truncate_enc:
                        enc = enc[:max_enc_len]
                    else:
                        continue
                dec = tokenize(pair[1])
                if max_dec_len and len(dec) > max_dec_len:
                    if truncate_dec:
                        dec = dec[:max_dec_len]
                    else:
                        continue
                self.pairs.append((enc, dec))
        print("%d pairs." % len(self.pairs))

    def build_vocab(self, embed_file=None):
        word_counts = Counter()
        count_words(word_counts, [enc + dec for enc, dec in self.pairs])
        vocab = Vocab()
        vocab.load_embeddings(embed_file)

        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab.add_words([word])

        return vocab


class SampleDataset(Dataset):
    def __init__(self, data_pair, vocab):
        self.src_sents = [x[0] for x in data_pair]
        self.trg_sents = [x[1] for x in data_pair]
        self.vocab = vocab
        self._len = len(data_pair)

    def __getitem__(self, index):
        x, oov = source2ids(self.src_sents[index], self.vocab)
        return {'x': [self.vocab.SOS] + x + [self.vocab.EOS],
                'OOV': oov,
                'len_OOV': len(oov),
                'y': [self.vocab.SOS] + abstract2ids(self.trg_sents[index], self.vocab, oov) + [self.vocab.EOS],
                'x_len': len(self.src_sents[index]),
                'y_len': len(self.trg_sents[index])
                }

    def __len__(self):
        return self._len


def collate_fn(batch):
    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)
    data_batch = sort_batch_by_len(batch)
    x = data_batch['x']
    x_max_length = max([len(t) for t in x])
    y = data_batch['y']
    y_max_length = max([len(t) for t in y])

    OOV = data_batch['OOV']
    len_OOV = torch.tensor(data_batch['len_OOV'])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch['x_len'])
    y_len = torch.tensor(data_batch['y_len'])

    return x_padded, y_padded, x_len, y_len, OOV, len_OOV


if __name__ == '__main__':
    pass
