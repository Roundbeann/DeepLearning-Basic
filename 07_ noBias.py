import numpy as np
from tqdm import trange
epoch = 100000
lr = 0.00001

# A 【14,3】
A = np.array( [
[3,2],
[4,3],
[5,3],
[7,4],
[8,5],
[9,7],
[12,7],
[3,1],
[4,1],
[5,2],
[7,3],
[8,3],
[9,4],
[12,5]
,])
# B 【3，1】
B = np.array([[-0.5],[0.5]])
# label【14，1】
label = np.array([[1],[1],[1],[1],[1],[1],[1],[-1],[-1],[-1],[-1],[-1],[-1],[-1]])
last_loss = 0
for i in trange(epoch):
    print(i)
    C = A @ B
    loss = (C - label)**2
    # G 【6 1】
    G = 2*(C - label)
    # A.T【4 6】 G【6 1】
    G_B= A.T @ G
    B =  B - G_B*lr

    print(loss.mean(),abs(last_loss-loss.mean()))
    if abs(loss.mean() - last_loss)<1e-7:
        break
    last_loss = loss.mean()

print(B)
for A_ele in A:
    print(A_ele @ B)
