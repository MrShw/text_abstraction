# step2
import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from seq2seq.utils.params_utils import get_params
from seq2seq.model_elements.model import Seq2Seq
from seq2seq.src.train_helper import train_model
from seq2seq.utils.config import *
from seq2seq.utils.word2vec_utils import get_vocab_from_model


# 训练主函数
def train(params):
    word_to_id, _ = get_vocab_from_model(vocab_path, reverse_vocab_path)  # 读取word_to_id训练
    params['vocab_size'] = len(word_to_id)  # 动态添加字典大小参数
    print("Building the model ...")
    model = Seq2Seq(params)  # 构建模型

    print('开始训练模型')
    train_model(model, word_to_id, params)  # 训练模型


if __name__ == '__main__':
    params = get_params()
    train(params)
