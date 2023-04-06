import numpy as np
import pandas as pd
import os
import sys
import re
import jieba
from seq2seq.utils.config import *
from seq2seq.utils import *
from seq2seq.utils.word2vec_utils import *
from seq2seq.utils.multi_proc_utils import *

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)


def get_max_len(data):
    # data: train_df['Question']
    max_lens = data.apply(lambda x: x.count(' ') + 1)

    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def transform_data(sentence, word_to_id):
    words = sentence.split(' ')     # sentence: 'word1 word2 word3 ...'  ->  [index1, index2, index3 ...]
    ids = [word_to_id[w] if w in word_to_id else word_to_id['<UNK>'] for w in words]

    return ids


def pad_proc(sentence, max_len, word_to_id):
    words = sentence.strip().split(' ')
    words = words[:max_len]

    sentence = [w if w in word_to_id else '<UNK>' for w in words]
    sentence = ['<START>'] + sentence + ['<STOP>']
    sentence = sentence + ['<PAD>'] * (max_len - len(words))

    return ' '.join(sentence)


def load_stop_words(stop_word_path):
    f = open(stop_word_path, 'r', encoding='utf-8')
    stop_words = f.readlines()
    stop_words = [stop_word.strip() for stop_word in stop_words]

    return stop_words


def clean_sentence(sentence):
    if isinstance(sentence, str):
        sentence = re.sub(r"\D(\d\.)\D", "", sentence)
        sentence = re.sub(r"[(（]进口[)）]|\(海外\)", "", sentence)
        sentence = re.sub(r"[^，！？。\.\-\u4e00-\u9fa5_a-zA-Z0-9]", "", sentence)
        sentence = sentence.replace(",", "，")
        sentence = sentence.replace("!", "！")
        sentence = sentence.replace("?", "？")
        sentence = re.sub(r"车主说|技师说|语音|图片|你好|您好", "", sentence)

        return sentence
    else:
        return ''


def filter_stopwords(seg_list):
    stop_words = load_stop_words(stop_words_path)
    words = [word for word in seg_list if word]     # seg_list: [word1 ,word2 .......]

    return [word for word in words if word not in stop_words]


def sentence_proc(sentence):
    sentence = clean_sentence(sentence)
    words = jieba.cut(sentence)
    words = filter_stopwords(words)

    return ' '.join(words)


def sentences_proc(df):
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)

    if 'Report' in df.columns:
        df['Report'] = df['Report'].apply(sentence_proc)

    return df

def load_train_dataset(max_enc_len=300, max_dec_len=50):

    train_X = np.load(train_x_path)
    train_Y = np.load(train_y_path)
    train_X = train_X[:, :max_enc_len]
    train_Y = train_Y[:, :max_dec_len]

    return train_X, train_Y


def load_test_dataset(max_enc_len=300):
    test_X = np.load(test_x_path)
    test_X = test_X[:, :max_enc_len]

    return test_X


def build_dataset(train_raw_data_path, test_raw_data_path):

    print('1. 加载原始数据')
    print("train_raw_data_path from",train_raw_data_path)
    train_df = pd.read_csv(train_raw_data_path, engine='python', encoding='utf-8')
    test_df = pd.read_csv(test_raw_data_path, engine='python', encoding='utf-8')
    print('原始训练集行数 {}, 测试集行数 {}'.format(len(train_df), len(test_df)))
    print('\n')

    print('2. 空值去除（对于一行数据，任意列只要有空值就去掉该行）')
    train_df.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    test_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)
    print('空值去除后训练集行数 {}, 测试集行数 {}'.format(len(train_df), len(test_df)))
    print('\n')

    print('3. 多线程, 批量数据预处理(对每个句子执行sentence_proc, 清除无用词, 分词, 过滤停用词, 再用空格拼接为一个字符串)')
    train_df = parallelize(train_df, sentences_proc)
    test_df = parallelize(test_df, sentences_proc)
    print('\n')
    print('sentences_proc has done!')

    print('4. 合并训练测试集, 用于构造映射字典word_to_id')
    # 按行堆积
    train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    # 按列堆积
    merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
    print(f'训练集行数{len(train_df)}, 测试集行数{len(test_df)}, 合并数据集行数{len(merged_df)}')
    print('\n')

    print('5. 保存分割处理好的train_seg_data.csv, test_set_data.csv')
    train_df = train_df.drop(['merged'], axis=1)
    test_df = test_df.drop(['merged'], axis=1)
    train_df.to_csv(train_seg_path, index=None, header=True)
    test_df.to_csv(test_seg_path, index=None, header=True)
    print('The csv_file has saved!')
    print('\n')

    print('6. 保存合并数据merged_seg_data.csv, 用于构造映射字典word_to_id')
    merged_df.to_csv(merged_seg_path, index=None, header=False)
    print('The word_to_vector file has saved!')
    print('\n')

    word_to_id = {}
    count = 0

    with open(merged_seg_path, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            line = line.strip().split(' ')
            for w in line:
                if w not in word_to_id:
                    word_to_id[w] = 1
                    count += 1
                else:
                    word_to_id[w] += 1

    print('总体单词总数count=', count)
    print('\n')

    res_dict = {}
    number = 0
    for w, i in word_to_id.items():
        if i >= 5:
            res_dict[w] = i
            number += 1

    print('进入到字典中的单词总数number=', number)
    print('合并数据集的字典构造完毕, word_to_id容量: ', len(res_dict))
    print('\n')

    word_to_id = {}
    count = 0
    for w, i in res_dict.items():
        if w not in word_to_id:
            word_to_id[w] = count
            count += 1

    print('最终构造完毕字典, word_to_id容量=', len(word_to_id))
    print('count=', count)

    print("8. 将Question和Dialogue用空格连接作为模型输入形成train_df['X']")
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    print('\n')

    print('9. 填充<START>, <STOP>, <UNK> 和 <PAD>, 使数据变为等长')

    train_x_max_len = get_max_len(train_df['X'])
    test_x_max_len = get_max_len(test_df['X'])
    train_y_max_len = get_max_len(train_df['Report'])

    print('填充前训练集样本的最大长度为: ', train_x_max_len)
    print('填充前测试集样本的最大长度为: ', test_x_max_len)
    print('填充前训练集标签的最大长度为: ', train_y_max_len)

    x_max_len = max(train_x_max_len, test_x_max_len)

    # train_df['X'] = train_df['X'].apply(lambda x: pad_proc(x, x_max_len, vocab))
    print('训练集X填充PAD, START, STOP, UNK处理中...')
    train_df['X'] = train_df['X'].apply(lambda x: pad_proc(x, x_max_len, word_to_id))

    print('测试集X填充PAD, START, STOP, UNK处理中...')
    test_df['X'] = test_df['X'].apply(lambda x: pad_proc(x, x_max_len, word_to_id))

    print('训练集Y填充PAD, START, STOP, UNK处理中...')
    train_df['Y'] = train_df['Report'].apply(lambda x: pad_proc(x, train_y_max_len, word_to_id))
    print('\n')

    print('10. 保存填充<START>, <STOP>, <UNK> 和 <PAD>后的X和Y')
    train_df['X'].to_csv(train_x_pad_path, index=None, header=False)
    train_df['Y'].to_csv(train_y_pad_path, index=None, header=False)
    test_df['X'].to_csv(test_x_pad_path, index=None, header=False)
    print('填充后的三个文件保存完毕!')
    print('\n')

    word_to_id = {}
    count = 0

    with open(train_x_pad_path, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            line = line.strip().split(' ')
            for w in line:
                if w not in word_to_id:
                    word_to_id[w] = count
                    count += 1

    print('训练集X字典构造完毕, word_to_id容量: ', len(word_to_id))

    with open(train_y_pad_path, 'r', encoding='utf-8') as f2:
        for line in f2.readlines():
            line = line.strip().split(' ')
            for w in line:
                if w not in word_to_id:
                    word_to_id[w] = count
                    count += 1

    print('训练集Y字典构造完毕, word_to_id容量: ', len(word_to_id))

    with open(test_x_pad_path, 'r', encoding='utf-8') as f3:
        for line in f3.readlines():
            line = line.strip().split(' ')
            for w in line:
                if w not in word_to_id:
                    word_to_id[w] = count
                    count += 1

    print('测试集X字典构造完毕, word_to_id容量: ', len(word_to_id))
    print('单词总数量count= ', count)

    id_to_word = {}
    for w, i in word_to_id.items():
        id_to_word[i] = w

    print('逆向字典构造完毕, id_to_word容量: ', len(id_to_word))
    print('\n')

    print('12. 更新vocab并保存')
    save_vocab_as_txt(vocab_path, word_to_id)
    save_vocab_as_txt(reverse_vocab_path, id_to_word)
    print('字典映射器word_to_id, id_to_word保存完毕!')
    print('\n')

    print('13. 数据集转换 将词转换成索引[<START> 方向机 重 ...] -> [32800, 403, 986, 246, 231]')
    print('训练集X执行transform_data中......')
    train_ids_x = train_df['X'].apply(lambda x: transform_data(x, word_to_id))
    print('训练集Y执行transform_data中......')
    train_ids_y = train_df['Y'].apply(lambda x: transform_data(x, word_to_id))
    print('测试集X执行transform_data中......')
    test_ids_x = test_df['X'].apply(lambda x: transform_data(x, word_to_id))
    print('\n')

    # 将索引列表转换成矩阵 [32800, 403, 986, 246, 231] --> array([[32800, 403, 986, 246, 231], ...])
    print('14. 数据转换成numpy数组(需等长)')
    train_X = np.array(train_ids_x.tolist())
    train_Y = np.array(train_ids_y.tolist())
    test_X = np.array(test_ids_x.tolist())
    print('转换为numpy数组的形状如下: \ntrain_X的shape为: ', train_X.shape, '\ntrain_Y的shape为: ', train_Y.shape, '\ntest_X的shape为: ', test_X.shape)
    print('\n')

    print('15. 保存数据......')
    np.save(train_x_path, train_X)
    np.save(train_y_path, train_Y)
    np.save(test_x_path, test_X)
    print('\n')
    print('数据集构造完毕, 存储于seq2seq/data/目录下.')


if __name__ == '__main__':
    # sentence = '技师说：你好！以前也出现过该故障吗？|技师说：缸压多少有没有测量一下?|车主说：没有过|车主说：没测缸\
    # 压 | 技师说：测量一下缸压\
    # 看一四缸缸压是否偏低 | 车主说：用电脑测，只是14缸缺火 | 车主说：[语音] | 车主说：[语音] | 技师\
    # 说：点火线圈\
    # 火花塞\
    # 喷油嘴不用干活\
    # 直接和二三缸对倒一下\
    # 跑一段在测量一下故障码进行排除 | 车主说：[语音] | 车主 > 说：[语音] | 车主说：[语音] | 车主说：[\
    #                                                                                                语音] | 车主说：师傅还在吗 | 技师说：调一下喷油嘴\
    # 测一下缸压\
    # 都正常则为发动机\
    # 电脑板问题 | 车主说：[语音] | 车主说：[语音] | 车主说：[语音] | 技师说：这个影响不大的 | 技师说：缸压八个以上正常 | 车主说\
    # ：[语音] | 技师说：所以说让你测量缸压\
    # 只要缸压正常则没有问题 | 车主说：[语音] | 车主说：[语音] | 技师说：可以点击头像\
    # 关注我\
    # 有什么问题随时询问\
    # 一定真诚用心为你解决 | 车主说：师傅，谢谢了 | 技师说：不用客气\
    # '
    #
    # res = sentence_proc(sentence)
    # print('res=', res)

    build_dataset(train_raw_data_path, test_raw_data_path)
