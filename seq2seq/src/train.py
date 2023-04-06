import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from seq2seq.utils.params_utils import get_params
from seq2seq.model_elements.model import Seq2Seq
from seq2seq.src.train_helper import train_model
from seq2seq.utils.config import *
from seq2seq.utils.word2vec_utils import get_vocab_from_model


def train(params):
    word_to_id, _ = get_vocab_from_model(vocab_path, reverse_vocab_path)
    params['vocab_size'] = len(word_to_id)
    print("Building the model ...")
    model = Seq2Seq(params)

    print('开始训练模型')
    train_model(model, word_to_id, params)


if __name__ == '__main__':
    params = get_params()
    train(params)
