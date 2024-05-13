import numpy as np
from houseData  import houseData

from tqdm import trange
houseData = np.array(houseData)

epoch = 15
lr = 0.001

A = houseData[:,1:]
label = houseData[:,0]

# z-score归一化方式
# 【label】 220，1
label_mean = label.mean()
label_std = label.std()
label = (label - label_mean)/label_std
label = label.reshape(len(label),1)

# z-score归一化方式
# 【A】 220，6
A_mean = A.mean(axis = 0)
A_std = A.std(axis = 0)
A = (A - A_mean)/A_std

# A【220,6】
# B【6,1】
# label【220,1】
# C = A * B 【220,1】
B = np.array([[1],[0],[1],[0],[1],[0]])

for i in trange(epoch):
    # C 【220,1】
    C = A @ B
    # 计算
    loss = (C - label)**2
    singleLoss = loss.sum()/len(A)
    # G 【220 1】
    G = 2 * (C - label)
    # A.T【6 220】 G【220 1】
    G_B= A.T @ G
    B =  B - G_B*lr
    # print(f"loss\n{loss}\nB\n{B}\nG_B*lr\n{G_B*lr}")
    print(singleLoss)
print(B)

predata = np.array([3,2,1,106.64,18,2007])
process_predata = (predata-A_mean)/A_std

print(process_predata @ B * label_std +label_mean)
