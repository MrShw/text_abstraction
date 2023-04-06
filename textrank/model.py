import pandas as pd
import numpy as np
import re
from textrank4zh import TextRank4Keyword, TextRank4Sentence

df = pd.read_csv("./data/dev.csv")


def clean_sentence(sentence):
    sub_jishi = []
    sub = sentence.split('|')

    for i in range(len(sub)):
        if not sub[i].endswith('。'):
            sub[i] += '。'
        if sub[i].startswith('技师'):
            sub_jishi.append(sub[i])

    sentence = ''.join(sub_jishi)
    sentence = re.sub(r"\D(\d\.)\D", "", sentence)
    sentence = re.sub(r"车主说|技师说|语音|图片|呢|吧|哈|啊|啦", "", sentence)
    sentence = re.sub(r"[(（]进口[)）]|\(海外\)", "", sentence)
    sentence = re.sub(r"[^，！？。\.\-\u4e00-\u9fa5_a-zA-Z0-9]", "", sentence)
    # sentence = re.sub(r"你好|您好|您的|你的|你|您|朋友|不客气|了|的|放心吧", "", sentence)

    sentence = sentence.replace(",", "，")
    sentence = sentence.replace("!", "！")
    sentence = sentence.replace("?", "？")
    sentence = sentence.replace("？", "。")
    sentence = sentence.replace("！", "。")
    sentence = sentence.replace("，。", "。")
    sentence = sentence.replace("。。", "。")
    sentence = sentence.replace("。，", "。")

    if sentence.startswith('，'):
        sentence = sentence[1:]

    return sentence


if __name__ == '__main__':
    df = pd.read_csv('./data/dev.csv', engine='python', encoding='utf-8')
    texts = df['Dialogue'].tolist()

    for i in range(len(texts)):
        texts[i]=clean_sentence(texts[i])

    results = []
    tr4s = TextRank4Sentence()

    # 从每个样本语句中提取关键句
    for i in range(len(texts)):
        text = texts[i]
        tr4s.analyze(text=text, lower=True, source='all_filters')
        result = ''

        for item in tr4s.get_key_sentences(num=3, sentence_min_len=2):  # 获取重要性最高的3个句子，句子的长度最小等于2
            result += item.sentence
            result += '。'

        results.append(result)

        if (i + 1) % 100 == 0:
            print(i + 1, result)

    print('result length: ', len(results))

    df['Prediction'] = results
    df = df[['QID', 'Report', 'Prediction']]
    df.to_csv('./data/textrank_result_.csv', index=None, sep=',')
    df = pd.read_csv('./data/textrank_result_.csv', engine='python', encoding='utf-8')
    df = df.fillna('随时联系。')
    df.to_csv('./data/textrank_result_final_.csv', index=None, sep=',')


