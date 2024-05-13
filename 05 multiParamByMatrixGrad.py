import numpy as np
from tqdm import trange
epoch = 300000
lr = 0.00001

# label = 3x_1 + 4x_2 - 5x_3 - 3.5
# 真正的 w1 = 3, w2 = 4, w3 = -5, b = -3.5
A = np.array( [[1,2,3,1],[-1,2,3,1],[-1,5,2,1],[2,-3,-4,1],[-2,-3,5,1],[2,3,-6,1]])
B = np.array([[1],[0],[1],[0]])
label = np.array([[-0.5],[-6.5],[10.5],[17.5],[-39.5],[51.5]])

for i in trange(epoch):
    C = A @ B
    loss = (C - label)**2
    # G 【6 1】
    G = 2*(C - label)
    # A.T【4 6】 G【6 1】
    G_B= A.T @ G
    B =  B - G_B*lr
print(B)
