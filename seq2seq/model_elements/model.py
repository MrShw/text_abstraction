import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import torch
import torch.nn as nn
from seq2seq.model_elements.layers import Encoder, Attention, Decoder
from seq2seq.utils.config import *
from seq2seq.utils.word2vec_utils import get_vocab_from_model


# 构建完整的seq2seq模型
class Seq2Seq(nn.Module):
    def __init__(self, params):
        super(Seq2Seq, self).__init__()
        self.params = params

        self.encoder = Encoder(params['vocab_size'], params['embed_size'],
                               params['enc_units'], params['batch_size'])
        self.attention = Attention(params['enc_units'], params['dec_units'],
                                   params['attn_units'])
        self.decoder = Decoder(params['vocab_size'], params['embed_size'],
                               params['dec_units'], params['batch_size'])

    def forward(self, dec_input, dec_hidden, enc_output, dec_target):
        predictions = []
        context_vector, attention_weights = self.attention(dec_hidden, enc_output)

        for t in range(dec_target.shape[1]):
            pred, dec_hidden = self.decoder(dec_input, context_vector)
            context_vector, attention_weights = self.attention(dec_hidden, enc_output)
            dec_input = dec_target[:, t].unsqueeze(1)
            predictions.append(pred)

        return torch.stack(predictions, 1), dec_hidden

# if __name__ == '__main__':
#     word_to_id, id_to_word = get_vocab_from_model(vocab_path, reverse_vocab_path)
#
#     vocab_size = len(word_to_id)
#     batch_size = 64
#     input_seq_len = 300
#
#     # 模拟测试参数
#     params = {"vocab_size": vocab_size, "embed_size": 500, "enc_units": 512,
#               "attn_units": 20, "dec_units": 512,"batch_size": batch_size}
#
#     model = Seq2Seq(params)
#
#     sample_input_batch = torch.ones((batch_size, input_seq_len), dtype=torch.long)
#     sample_hidden = model.encoder.initialize_hidden_state()
#
#     sample_output, sample_hidden = model.encoder(sample_input_batch, sample_hidden)
#
#     print('Encoder output shape: (batch_size, enc_seq_len, enc_units) {}'.format(sample_output.shape))
#     print('Encoder Hidden state shape: (batch_size, enc_units) {}'.format(sample_hidden.shape))
#
#     context_vector, attention_weights = model.attention(sample_hidden, sample_output)
#
#     print("Attention context_vector shape: (batch_size, enc_units) {}".format(context_vector.shape))
#     print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
#
#     dec_input = torch.ones((batch_size, 1), dtype=torch.long)
#     sample_decoder_output, _, = model.decoder(dec_input, context_vector)
#
#     print('Decoder output shape: (batch_size, vocab_size) {}'.format(sample_decoder_output.shape))