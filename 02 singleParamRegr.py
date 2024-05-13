from tqdm import trange

epoch = 1000
lr = 0.001

# label = 3x^2 + 4x - 5
# 真正的 w1 = 3, w2 = 4, b = -5
x = [1,2,0,-1]
label = [2,15,-5,-6]

w1 = 1
w2 = 2
b = 0

for i in trange(10000):
    loss = 0.
    grad_w1=0.
    grad_w2=0.
    grad_b=0.
    for i in range(len(x)):
        pre = x[i]**2*w1 + x[i]*w2 + b
        loss += (pre - label[i])**2
        # loss = ( c[i]**2*w1 + c[i]*w2 + b - label)**2
        grad_w1 += 2 * ( pre - label[i])*x[i]**2
        grad_w2 += 2 * ( pre - label[i])*x[i]
        grad_b += 2 * ( pre - label[i])
    grad_w1 /=  len(x)
    grad_w2 /=  len(x)
    grad_b /=  len(x)
    w1 -= grad_w1 * lr
    w2 -= grad_w2 * lr
    b -= grad_b * lr

print(w1,w2,b)