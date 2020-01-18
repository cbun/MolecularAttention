import pandas as pd
import numpy as np
from itertools import product

dropout_rate = np.linspace(0, 0.4, 5, dtype=np.int32)
batch_size = np.linspace(32, 256, 5, dtype=np.int32)
intermediate_rep = np.linspace(64, 512, 5, dtype=np.int32)
nheads = 2 ** np.linspace(0, 10, 5, dtype=np.int32)
lr = 10.0 ** np.random.randint(-6, -2, 5, dtype=np.int32)
cylic = [True, False]

print(len(list(product(*[dropout_rate, batch_size, intermediate_rep, nheads, lr, cylic]))))