# step2
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool

cores = cpu_count()
partitions = cores


def parallelize(df, func):
    data_split = np.array_split(df, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    pool.join()

    return data
