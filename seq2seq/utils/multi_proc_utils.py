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

    pool.join()

    return data
