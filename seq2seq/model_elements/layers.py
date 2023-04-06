# 实现编码器类Encoder
# 实现注意力类Attention
# 实现解码器类Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from seq2seq.utils.config import *
from seq2seq.utils.word2vec_utils import get_vocab_from_model


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=enc_units, num_layers=1, batch_first=True)

    def forward(self, x, h0):
        x = self.embedding(x)
        output, hn = self.gru(x, h0)

        return output, hn.transpose(1, 0)

    def initialize_hidden_state(self):

        return torch.zeros(1, self.batch_size, self.enc_units)


class Attention(nn.Module):
    def __init__(self, enc_units, dec_units, attn_units):
        super(Attention, self).__init__()
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.attn_units = attn_units

        self.w1 = nn.Linear(enc_units, attn_units)
        self.w2 = nn.Linear(dec_units, attn_units)
        self.v = nn.Linear(attn_units, 1)

    def forward(self, query, value):

        score = self.v(torch.tanh(self.w1(value) + self.w2(query)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * value
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim + dec_units,
                          hidden_size=dec_units,
                          num_layers=1,
                          batch_first=True)

        self.fc = nn.Linear(dec_units, vocab_size)

    def forward(self, x, context_vector):

        x = self.embedding(x)
        x = torch.cat([torch.unsqueeze(context_vector, 1), x], dim=-1)
        output, hn = self.gru(x)
        output = output.squeeze(1)
        prediction = self.fc(output)

        return prediction, hn.transpose(1, 0)

# if __name__ == '__main__':
#     word_to_id, id_to_word = get_vocab_from_model(vocab_path, reverse_vocab_path)
#     vocab_size = len(word_to_id)
#     print('vocab_size: ', vocab_size)
#
#     # 测试用参数
#     EXAMPLE_INPUT_SEQUENCE_LEN = 300
#     BATCH_SIZE = 64
#     EMBEDDING_DIM = 500
#     GRU_UNITS = 512
#     ATTENTION_UNITS = 20
#
#     encoder = Encoder(vocab_size, EMBEDDING_DIM, GRU_UNITS, BATCH_SIZE)
#
#     input0 = torch.ones((BATCH_SIZE, EXAMPLE_INPUT_SEQUENCE_LEN), dtype=torch.long)
#     h0 = encoder.initialize_hidden_state()
#     output, hn = encoder(input0, h0)
#     print(output.shape)
#     print(hn.shape)
