# step5
import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import numpy as np
import time
from collections import Counter
import heapq
import random
import pathlib
from pgn_attention.utils import config1
from pgn_attention.utils import config
import torch


# 函数耗时计量函数，修饰器
def timer(module):
    def wrapper(func):
        # func: 一个函数名, 下面的计时函数就是计算这个func函数的耗时
        def cal_time( *args, **kwargs):
            t1 = time.time()
            res = func( *args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time:.6f} secs used for: ', module)
            return res
        return cal_time
    return wrapper


def simple_tokenizer(text):
    return text.split()


def count_words(counter, text):
    for sentence in text:
        for word in sentence:
            counter[word] += 1


# 对一个批次batch_size个样本, 按照x_len字段长短进行排序, 并返回排序后的结果
def sort_batch_by_len(data_batch):
    res = {'x': [],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': []}

    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()
    data_batch = {name: [_tensor[i] for i in sorted_indices] for name, _tensor in res.items()}

    return data_batch


# 原始文本映射成ids张量
def source2ids(source_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.UNK
    for w in source_words:
        i = vocab[w]
        if i == unk_id:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)

    return ids, oovs

# 摘要文本映射成数字化ids张量
def abstract2ids(abstract_words, vocab, source_oovs):
    ids = []
    unk_id = vocab.UNK
    for w in abstract_words:
        i = vocab[w]
        if i == unk_id:
            if w in source_oovs:
                vocab_idx = vocab.size() + source_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)

    return ids


# 将输出张量ids结果映射成自然语言文本
def outputids2words(id_list, source_oovs, vocab):
    words = []
    for i in id_list:
        try:
            w = vocab.index2word[i]
        except IndexError:
            assert_msg = "Error: 无法在词典中找到该ID值."
            assert source_oovs is not None, assert_msg
            source_oov_idx = i - vocab.size()
            try:
                w = source_oovs[source_oov_idx]
            except ValueError:
                raise ValueError(f'Error: 模型生成的ID: {i}, 原始文本中的OOV ID: {source_oov_idx} \
                                  但是当前样本中只有{source_oovs}个OOVs')
        words.append(w)

    return ' '.join(words)


# 创建小顶堆, 包含k个点的特殊二叉树, 始终保持二叉树中最小的值在root根节点
def add2heap(heap, item, k):
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)


# 将文本张量中所有OOV单词的id, 全部替换成<UNK>对应的id
def replace_oovs(in_tensor, vocab):
    oov_token = torch.full(in_tensor.shape, vocab.UNK, dtype=torch.long).to(config.DEVICE)
    out_tensor = torch.where(in_tensor > len(vocab) - 1, oov_token, in_tensor)

    return out_tensor


# 获取模型训练中若干超参数信息
def config_info(config):
    info = 'model_name = {}, pointer = {}, coverage = {}, fine_tune = {}, scheduled_sampling = {}, weight_tying = {},' + 'source = {}  '
    return (info.format(config.model_name, config.pointer, config.coverage, config.fine_tune, config.scheduled_sampling, config.weight_tying, config.source))


class ScheduledSampler():
    def __init__(self, phases):
        self.phases = phases
        self.scheduled_probs = [i / (self.phases - 1) for i in range(self.phases)]

    def teacher_forcing(self, phase):
        # According to a certain probability to choose whether to execute teacher_forcing
        sampling_prob = random.random()
        if sampling_prob >= self.scheduled_probs[phase]:
            return True
        else:
            return False


if __name__ == '__main__':
    @timer(module='test a demo program')
    def test():
        s = 0
        for i in range(101):
            s += i
        print('s = ', s)
    test()

    sentence = "技师说：你好！以前也出现过该故障吗？|技师说：缸压多少有没有测量一下?|车主说：没有过|车主说：没测缸压 | 技师说：测量一下缸压看一四缸缸压是否偏低 | 车主说：用电脑测，只是14缸缺火 | " \
               "车主说：[语音] | 车主说：[语音] | 技师说：点火线圈火花塞喷油嘴不用干活直接和二三缸对倒一下跑一段在测量一下故障码进行排除 | 车主说：[语音] | 车主 > 说：[语音] | 车主说：[语音] " \
               "| 车主说：[语音] | 车主说：师傅还在吗 | 技师说：调一下喷油嘴测一下缸压都正常则为发动机电脑板问题 | 车主说：[语音] | 车主说：[语音] | 车主说：[语音] | 技师说：这个影响不大的 " \
               "| 技师说：缸压八个以上正常 | 车主说：[语音] | 技师说：所以说让你测量缸压只要缸压正常则没有问题 | 车主说：[语音] | 车主说：[语音] | 技师说：可以点击头像关注我\ " \
               "有什么问题随时询问一定真诚用心为你解决 | 车主说：师傅，谢谢了 | 技师说：不用客气"

    res = simple_tokenizer(sentence)
    print('res=', res)
    print('res length = ', len(res))


    counter = Counter()
    text = ['以前也出现过该故障吗？缸压多少有没有测量一下?',
            '14缸缺火点火线圈火花塞喷油嘴不用干活直接和二三缸对倒一下跑一段在测量一下故障码']
    count_words(counter, text)
    for w, c in counter.most_common(100):
        print(w, ': ', c)

    from pgn_attention.utils import config
    res = config_info(config)
    print(res)