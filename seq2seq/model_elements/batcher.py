import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from seq2seq.utils.data_loader import load_train_dataset, load_test_dataset


def train_batch_generator(batch_size, max_enc_len=300, max_dec_len=50, sample_num=None):
    train_X, train_Y = load_train_dataset(max_enc_len, max_dec_len)

    if sample_num:
        train_X = train_X[:sample_num]
        train_Y = train_Y[:sample_num]
    x_data = torch.from_numpy(train_X)
    y_data = torch.from_numpy(train_Y)

    dataset = TensorDataset(x_data, y_data)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    steps_per_epoch = len(train_X) // batch_size

    return dataset, steps_per_epoch

def test_batch_generator(batch_size, max_enc_len=300):

    test_X = load_test_dataset(max_enc_len)
    x_data = torch.from_numpy(test_X)
    dataset = TensorDataset(x_data)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    steps_per_epoch = len(test_X) // batch_size

    return dataset, steps_per_epoch

# if __name__ == '__main__':
#     dataset1, length1 = train_batch_generator(64)
#     dataset2, length2 = test_batch_generator(64)
#     print(dataset1)
#     print(length1)
#     print(dataset2)
#     print(length2)