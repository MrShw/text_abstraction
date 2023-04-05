# step3
import random
import os
import sys
import torch
import jieba

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入项目的相关代码文件
from pgn_attention.utils import config
from pgn_attention.model_elements.model import PGN
from pgn_attention.utils.dataset import PairDataset
from pgn_attention.utils.func_utils import source2ids, outputids2words, timer, add2heap, replace_oovs


# 构建预测类
class Predict():
    @timer(module='initalize predicter')
    def __init__(self):
        self.DEVICE = config.DEVICE

        dataset = PairDataset(config.train_data_path,
                              max_enc_len=config.max_enc_len,
                              max_dec_len=config.max_dec_len,
                              truncate_enc=config.truncate_enc,
                              truncate_dec=config.truncate_dec)

        self.vocab = dataset.build_vocab(embed_file=config.embed_file)

        self.model = PGN(self.vocab)
        self.stop_word = list(set([self.vocab[x.strip()] for x in open(config.stop_word_file).readlines()]))
        self.model.load_state_dict(torch.load(config.model_save_path,map_location=torch.device('cpu')))
        self.model.to(self.DEVICE)

    def greedy_search(self, x, max_sum_len, len_oovs, x_padding_masks):
        encoder_output, encoder_states = self.model.encoder(replace_oovs(x, self.vocab))
        # 用encoder的hidden state初始化decoder的hidden state
        decoder_states = self.model.reduce_state(encoder_states)

        # 利用SOS作为解码器的初始化输入字符
        x_t = torch.ones(1) * self.vocab.SOS
        x_t = x_t.to(self.DEVICE, dtype=torch.int64)
        summary = [self.vocab.SOS]

        # 循环解码, 最多解码max_sum_len步
        while int(x_t.item()) != (self.vocab.EOS) and len(summary) < max_sum_len:
            context_vector, attention_weights = self.model.attention(decoder_states,
                                                                     encoder_output,
                                                                     x_padding_masks)

            p_vocab, decoder_states, p_gen = self.model.decoder(x_t.unsqueeze(1),
                                                                decoder_states,
                                                                context_vector)

            final_dist = self.model.get_final_distribution(x, p_gen, p_vocab,
                                                           attention_weights,
                                                           torch.max(len_oovs))

            # 以贪心解码策略预测字符
            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
            decoder_word_idx = x_t.item()

            # 将预测的字符添加进结果摘要中
            summary.append(decoder_word_idx)
            x_t = replace_oovs(x_t, self.vocab)

        return summary

    @timer(module='doing prediction')
    def predict(self, text, tokenize=True):
        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))
        x, oov = source2ids(text, self.vocab)
        x = torch.tensor(x).to(self.DEVICE)
        len_oovs = torch.tensor([len(oov)]).to(self.DEVICE)
        x_padding_masks = torch.ne(x, 0).byte().float()

        # 利用贪心解码函数得到摘要结果
        summary = self.greedy_search(x.unsqueeze(0),
                                     max_sum_len=config.max_dec_steps,
                                     len_oovs=len_oovs,
                                     x_padding_masks=x_padding_masks)

        # 将得到的摘要数字化张量转换成自然语言文本
        summary = outputids2words(summary, oov, self.vocab)

        # 删除掉特殊字符<SOS>和<EOS>
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip()


if __name__ == "__main__":
    print('实例化Predict对象, 构建dataset和vocab......')
    pred = Predict()
    print('vocab_size: ', len(pred.vocab))
    # Randomly pick a sample in test set to predict.
    with open(config.val_data_path, 'r') as test:
        picked = random.choice(list(test))
        source, ref = picked.strip().split('<SEP>')

    print('source: ', source, '\n')
    print('---------------------------------------------------------------')
    print('ref: ', ref, '\n')
    print('---------------------------------------------------------------')
    greedy_prediction = pred.predict(source.split())
    print('greedy: ', greedy_prediction, '\n')
