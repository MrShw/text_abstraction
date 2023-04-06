import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import torch
import torch.nn as nn
import time
import pandas as pd
from seq2seq.model_elements.model import Seq2Seq
from seq2seq.src.test_helper import greedy_decode
from seq2seq.utils.data_loader import load_test_dataset
from seq2seq.utils.config import *
from seq2seq.utils.params_utils import get_params
from seq2seq.utils.word2vec_utils import get_vocab_from_model

def test(params):
    device = torch.device('cpu' if torch.backends.mps.is_available() else 'cpu')
    print(f'Model has passed to {device}...')
    print("创建字典")
    word_to_id, id_to_word = get_vocab_from_model(vocab_path, reverse_vocab_path)
    params['vocab_size'] = len(word_to_id)

    print("创建模型")
    model = Seq2Seq(params)

    MODEL_PATH = root_path + '/src/saved_model/' + 'model_19.pt'
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device=device)))
    print("模型加载完毕!")

    print("生成测试数据迭代器")
    test_x = load_test_dataset()
    print("开始解码......")
    results = greedy_decode(model, test_x, word_to_id, id_to_word, params)
    print("解码完毕, 开始保存结果......")
    results = list(map(lambda x: x.replace(" ", ""), results))
    save_predict_result(results)

    return results


def save_predict_result(results):
    print("读取原始测试数据...")
    test_df = pd.read_csv(test_raw_data_path)
    print("构建新的DataFrame并保存文件...")
    test_df['Prediction'] = results
    test_df = test_df[['QID', 'Prediction']]
    test_df.to_csv(get_result_filename(), index=None, sep=',')
    print("保存测试结果完毕!")


def get_result_filename():
    now_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    filename = 'seq2seq_' + now_time + '.csv'
    result_path = os.path.join(result_save_path, filename)

    return result_path


if __name__ == '__main__':
    params = get_params()
    results = test(params)
    print(results[:10])