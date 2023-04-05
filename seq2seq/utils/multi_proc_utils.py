# step2
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool

cores = cpu_count()
partitions = cores


# 并行处理函数
def parallelize(df, func):
    data_split = np.array_split(df, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    # close后不会有新的进程加入到pool中，join函数等待所欲偶子进程结束后退出
    pool.join()

    return data
