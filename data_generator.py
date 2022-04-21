import numpy as np
import torch
np.random.seed(2)

T = 20
L = 1000 #点数
N = 100 #条数

x = np.empty((N, L), 'int64') # 10*100
x[:] = np.array(range(L)) 
x[:] += np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')

# x = np.empty((N, L), 'float64')
# x[:] =np.array(range(L)) 
# x[:] += np.random.randint(-4 * T, 4 * T, N).reshape(N, 1) 
# y = np.empty((N, L), 'float64')
# y[:] = np.random.uniform(-0.1,0.1,size=(N,L)) 

data = np.sin(x / 1.0 / T).astype('float64')
# data += y[:]

torch.save(data, open('train_raw_data.pt', 'wb'))