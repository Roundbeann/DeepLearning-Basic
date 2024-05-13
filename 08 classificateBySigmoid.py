# P = X @ k
# pre = sig(P)
# 分类 loss = label * log(pre) + (1-label) * log(1-pre)
# loss 对 k的导数
# 上式看起来是两项，实际上是一项 因为label要么为 0，要么为 1
# G = loss 对 P求导，得到

# A * B = C
# loss 对 C 的导数 为 G
# Loss 对 A 的导数 为 G * B.T
# loss 对 B 的导数 为 A.T * G

# 在此，要求 loss 对 k 的导数，首先求 loss 对 loss 对 P 的导数 G，再求 P 对 k 的导数
# P 对 k 的导数可以通过矩阵求导 可就是grad_k = X.T * G 求得

# loss 对 P 的导数 G 根据链式法则可以化为 loss 对 pre的导数乘以 pre对P的导数 结果为 C * (pre - Label)

# 回归 loss = (label - pre)**2

import numpy as np
from tqdm import trange
epoch = 100000
lr = 0.0001

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# A 【14,3】
X = np.array( [
[3,2,1],[4,3,1],[5,3,1],[7,4,1],[8,5,1],[9,7,1],[12,7,1],[3,1,1],[4,1,1],[5,2,1],[7,3,1],[8,3,1],[9,4,1],[12,5,1],])
# B 【3，1】
k = np.array([[-0.5],[0.5],[1]])
# label【14，1】
label = np.array([[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0]])

for i in trange(epoch):
    print(i)
    # X 【14，3】 k【3，1】 p 【14，1】
    p = X @ k

    # 分类与回归的不同点仅在于该步骤:加入了非线性因素sigmoid
    # pre 【14，1】
    pre = sigmoid(p)

    # loss = crossEntropy ...
    loss = - (label * np.log(pre) + (1-label) * np.log(1-pre))
    single_loss = loss.mean()
    # # G 【6 1】
    G = (pre - label)
    # # A.T【4 6】 G【6 1】
    G_k= X.T @ G
    k =  k - G_k*lr
    print(single_loss)

print(k)
for X_ele in X:
    print(sigmoid(X_ele @ k))
