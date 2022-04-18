import numpy as np
import torch
np.random.seed(2)

T = 20
L = 1000 #lie
N = 100 #hang

x = np.empty((N, L), 'int64') # 10*100
x[:] = np.array(range(L)) 
x[:] += np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')
torch.save(data, open('traindata.pt', 'wb'))