import torch
import torch.nn as nn
import torch.nn.functional as F
from pgn_coverage.utils import config


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_drop=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, dropout=rnn_drop, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        self.Wh = nn.Linear(2 * hidden_units, 2 * hidden_units, bias=False)
        self.Ws = nn.Linear(2 * hidden_units, 2 * hidden_units)
        self.wc = nn.Linear(1, 2 * hidden_units, bias=False)
        self.v = nn.Linear(2 * hidden_units, 1, bias=False)

    def forward(self, decoder_states, encoder_output, x_padding_masks, coverage_vector):
        h_dec, c_dec = decoder_states
        s_t = torch.cat([h_dec, c_dec], dim=2)
        s_t = s_t.transpose(0, 1)
        s_t = s_t.expand_as(encoder_output).contiguous()

        # 根据论文中的公式1来计算et
        encoder_features = self.Wh(encoder_output.contiguous())
        # Ws * s_t: (batch_size, seq_length, 2*hidden_units)
        decoder_features = self.Ws(s_t)

        # (batch_size, seq_length, 2*hidden_units)
        attn_inputs = encoder_features + decoder_features

        if config.coverage:
            coverage_features = self.wc(coverage_vector.unsqueeze(2))
            attn_inputs = attn_inputs + coverage_features

        # (batch_size, seq_length, 1)
        score = self.v(torch.tanh(attn_inputs))

        # (batch_size, seq_length)
        attention_weights = F.softmax(score, dim=1).squeeze(2)
        attention_weights = attention_weights * x_padding_masks
        normalization_factor = attention_weights.sum(1, keepdim=True)
        attention_weights = attention_weights / normalization_factor

        # (batch_size, 1, 2*hidden_units)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output)
        # (batch_size, 2*hidden_units)
        context_vector = context_vector.squeeze(1)

        if config.coverage:
            coverage_vector = coverage_vector + attention_weights

        return context_vector, attention_weights, coverage_vector


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, enc_hidden_size=None):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)

        if config.pointer:
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1)

    def forward(self, x_t, decoder_states, context_vector):
        decoder_emb = self.embedding(x_t)
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)
        decoder_output = decoder_output.view(-1, config.hidden_size)
        concat_vector = torch.cat([decoder_output, context_vector], dim=-1)
        # (batch_size, hidden_units)
        FF1_out = self.W1(concat_vector)
        # (batch_size, vocab_size)
        FF2_out = self.W2(FF1_out)
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)

        # 构造decoder state s_t.
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)

        p_gen = None
        if config.pointer:
            x_gen = torch.cat([context_vector, s_t.squeeze(0), decoder_emb.squeeze(1)], dim=-1)
            p_gen = torch.sigmoid(self.w_gen(x_gen))

        return p_vocab, decoder_states, p_gen


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)
