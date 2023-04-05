# step3
# 导入若干工具包
import re
import jieba
import pandas as pd
import numpy as np
import os
import sys

# 设置项目的root目录, 方便后续相关代码包的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入文本预处理的配置信息config1
from pgn_coverage.utils.config1 import *
# 导入多核CPU并行处理数据的函数
from pgn_coverage.utils.multi_proc_utils import *

# jieba载入自定义切词表
jieba.load_userdict(user_dict_path)


def pad_proc(sentence, max_len, word_to_id):
    words = sentence.strip().split(' ')
    words = words[:max_len]
    sentence = [w if w in word_to_id else '<UNK>' for w in words]
    sentence = ['<START>'] + sentence + ['<STOP>']
    sentence = sentence + ['<PAD>'] * (max_len - len(words))
    return ' '.join(sentence)


# 加载停用词(程序调用)
def load_stop_words(stop_word_path):
    f = open(stop_word_path, 'r', encoding='utf-8')
    stop_words = f.readlines()
    stop_words = [stop_word.strip() for stop_word in stop_words]

    return stop_words


# 加载停用词
stop_words = load_stop_words(stop_words_path)


# 清洗文本的函数，特殊符号去除(被sentence_proc调用)
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


# 过滤停用词的函数
def filter_stopwords(seg_list):
    stop_words = load_stop_words(stop_words_path)
    words = [word for word in seg_list if word]     # seg_list: 切好词的列表 [word1 ,word2 .......]

    return [word for word in words if word not in stop_words]



# 语句处理的函数，预处理模块(处理一条句子, 被sentences_proc调用)
def sentence_proc(sentence):
    sentence = clean_sentence(sentence)
    words = jieba.cut(sentence)
    words = filter_stopwords(words)

    return ' '.join(words)


# 语句处理的函数，预处理模块(处理一个句子列表, 对每个句子调用sentence_proc操作)
def sentences_proc(df):
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)

    if 'Report' in df.columns:
        df['Report'] = df['Report'].apply(sentence_proc)

    return df


# 用于数据加载+预处理(只需执行一次)
def build_dataset(train_raw_data_path, test_raw_data_path):
    print('1. 加载原始数据')
    print(train_raw_data_path)
    train_df = pd.read_csv(train_raw_data_path, engine='python', encoding='utf-8')
    test_df = pd.read_csv(test_raw_data_path, engine='python', encoding='utf-8')
    print(f'原始训练集行数 {len(train_df)}, 测试集行数 {len(test_df)}')
    print('\n')

    print('2. 空值去除（对于一行数据，任意列只要有空值就去掉该行）')
    train_df.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    test_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)
    print(f'空值去除后训练集行数 {len(train_df)}, 测试集行数 {len(test_df)}')
    print('\n')

    print('3. 多线程, 批量数据预处理(对每个句子执行sentence_proc，清除无用词，切词，过滤停用词，再用空格拼接为一个字符串)')
    train_df = parallelize(train_df, sentences_proc)
    test_df = parallelize(test_df, sentences_proc)
    print('\n')
    print('sentences_proc has done!')

    print('4. 合并训练测试集，用于训练词向量')
    # 新建一列，按行堆积
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    train_df['Y'] = train_df[['Report']]
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    print('5. 保存处理好的train_seg_data.csv、test_set_data.csv')
    # 把建立的列merged去掉，该列对于神经网络无用，只用于训练词向量
    train_df = train_df.drop(['Question'], axis=1)
    train_df = train_df.drop(['Dialogue'], axis=1)
    train_df = train_df.drop(['Brand'], axis=1)
    train_df = train_df.drop(['Model'], axis=1)
    train_df = train_df.drop(['Report'], axis=1)
    train_df = train_df.drop(['QID'], axis=1)
    test_df = test_df.drop(['Question'], axis=1)
    test_df = test_df.drop(['Dialogue'], axis=1)
    test_df = test_df.drop(['Brand'], axis=1)
    test_df = test_df.drop(['Model'], axis=1)
    test_df = test_df.drop(['QID'], axis=1)
    # 将处理后的数据存入持久化文件
    # train_df.to_csv(train_seg_path, index=None, header=True)
    test_df.to_csv(test_seg_path, index=None, header=False)
    train_df['data'] = train_df[['X', 'Y']].apply(lambda x: '<sep>'.join(x), axis=1)
    train_df = train_df.drop(['X'], axis=1)
    train_df = train_df.drop(['Y'], axis=1)
    train_df.to_csv(train_seg_path, index=None, header=False)
    # train_df = pd.read_csv(train_seg_path, header=None)
    # train_df = train_df.iloc[1:, :]
    # train_df.to_csv(train_seg_path, index=None, header=True)

    print('The csv_file has saved!')
    print('\n')

    print('6. 将预处理文件保存为预定格式的.txt文件')


if __name__ == '__main__':
    build_dataset(train_raw_data_path, test_raw_data_path)
    # 在data路径下, 多出了两个数据文件train_seg_data.csv和test_seg_data.csv

