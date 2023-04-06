import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgn_coverage.utils.func_utils import timer, replace_oovs
from pgn_coverage.utils.vocab import Vocab
from pgn_coverage.utils import config
from pgn_coverage.model_elements.layers import *


class PGN(nn.Module):
    def __init__(self, v):
        super(PGN, self).__init__()
        self.v = v
        self.DEVICE = config.DEVICE
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(len(v), config.embed_size, config.hidden_size)
        self.decoder = Decoder(len(v), config.embed_size, config.hidden_size)
        self.reduce_state = ReduceState()

    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights, max_oov):
        if not config.pointer:
            return p_vocab

        batch_size = x.size()[0]
        p_gen = torch.clamp(p_gen, 0.001, 0.999)
        p_vocab_weighted = p_gen * p_vocab
        # (batch_size, seq_len)
        attention_weighted = (1 - p_gen) * attention_weights

        # extended_size = len(self.v) + max_oovs
        extension = torch.zeros((batch_size, max_oov)).float().to(self.DEVICE)
        # (batch_size, extended_vocab_size)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)
        final_distribution = p_vocab_extended.scatter_add_(dim=1, index=x, src=attention_weighted)

        return final_distribution

    def forward(self, x, x_len, y, len_oovs, batch, num_batches, teacher_forcing):
        x_copy = replace_oovs(x, self.v)
        x_padding_masks = torch.ne(x, 0).byte().float()
        encoder_output, encoder_states = self.encoder(x_copy)
        decoder_states = self.reduce_state(encoder_states)

        coverage_vector = torch.zeros(x.size()).to(self.DEVICE)

        step_losses = []

        x_t = y[:, 0]
        for t in range(y.shape[1] - 1):

            if teacher_forcing:
                x_t = y[:, t]

            x_t = replace_oovs(x_t, self.v)
            y_t = y[:, t + 1]

            context_vector, attention_weights, coverage_vector = self.attention(decoder_states,
                                                                                encoder_output,
                                                                                x_padding_masks,
                                                                                coverage_vector)

            p_vocab, decoder_states, p_gen = self.decoder(x_t.unsqueeze(1), decoder_states, context_vector)

            final_dist = self.get_final_distribution(x, p_gen, p_vocab, attention_weights, torch.max(len_oovs))


            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)

            if not config.pointer:
                y_t = replace_oovs(y_t, self.v)
            target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1)

            mask = torch.ne(y_t, 0).byte()
            loss = -torch.log(target_probs + config.eps)

            if config.coverage:
                ct_min = torch.min(attention_weights, coverage_vector)
                cov_loss = torch.sum(ct_min, dim=1)
                loss = loss + config.LAMBDA * cov_loss

            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)

        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
        seq_len_mask = torch.ne(y, 0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)
        batch_loss = torch.mean(sample_losses / batch_seq_len)

        return batch_loss


if __name__ == '__main__':
    v = Vocab()
    model = PGN(v)
    print(model)
