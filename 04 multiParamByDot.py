from tqdm import trange
import numpy as np

epoch = 1000
lr = 0.00001

# label = 3x_1 + 4x_2 - 5x_3 - 3.5
# 真正的 w1 = 3, w2 = 4, w3 = -5, b = -3.5
x = np.array( [[1,2,3,1],[-1,2,3,1],[-1,5,2,1],[2,-3,-4,1],[-2,-3,5,1],[2,3,-6,1]])
label = [-0.5,-6.5,10.5,17.5,-39.5,51.5]

W = np.array([0.,0.,0.,0])


for i in trange(300000):
    loss = 0.
    grad_w1=0.
    grad_w2=0.
    grad_w3=0.
    grad_b=0.
    for i in range(len(x)):
        pre = np.dot(x[i],W).sum()
        loss += (pre - label[i])**2
        # loss = ( c[i]**2*w1 + c[i]*w2 + b - label)**2
        grad_w1 += 2 * (pre- label[i])*x[i][0]
        grad_w2 += 2 * (pre- label[i])*x[i][1]
        grad_w3 += 2 * (pre- label[i])*x[i][2]
        grad_b += 2 * (pre- label[i])
    grad_w1 /=  len(x)
    grad_w2 /=  len(x)
    grad_w3 /=  len(x)
    grad_b /=  len(x)
    W[0] -= grad_w1 * lr
    W[1] -= grad_w2 * lr
    W[2] -= grad_w3 * lr
    W[3] -= grad_b * lr

print(W)