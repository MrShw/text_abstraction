# step1
import os
import sys
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from pgn_coverage.utils.dataset import collate_fn
from pgn_coverage.utils import config


# 评估函数
def evaluate(model, val_data, epoch):
    print('validating')
    val_loss = []
    with torch.no_grad():
        DEVICE = config.DEVICE
        # 创建数据迭代器, pin_memory=True是对于GPU机器的优化设置
        # 为了PGN模型数据的特殊性, 传入自定义的collate_fn提供个性化服务
        val_dataloader = DataLoader(dataset=val_data,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=collate_fn)

        # 遍历测试集数据进行评估
        for batch, data in enumerate(tqdm(val_dataloader)):
            x, y, x_len, y_len, oov, len_oovs = data
            if config.is_cuda:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)
            total_num = len(val_dataloader)
            loss = model(x, x_len, y, len_oovs, batch=batch, num_batches=total_num, teacher_forcing=True)
            val_loss.append(loss.item())

    return np.mean(val_loss)    # 返回整个测试集的平均损失值

