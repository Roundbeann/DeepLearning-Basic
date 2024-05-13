from tqdm import trange
lr = 0.0001

# label = 5x^3 - 2x^2 + 3x - 4
# 真正的 w1 = 5, w2 = -2, w3 = 3 b = -4
x = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
label= [-368, -166, -58, -14, -4, 2, 34, 122, 296, 586]

w1 = 0
w2 = 0
w3 = 0
b = 0

for i in trange(300000):
    loss = 0.
    grad_w1=0.
    grad_w2=0.
    grad_w3 = 0.
    grad_b=0.
    for i in range(len(x)):
        pre = x[i]**3*w1 + x[i]**2*w2 + x[i]*w3 + b
        loss += (pre - label[i])**2
        # loss = ( x[i]**3*w1 + x[i]**2*w2 + x[i]*w3 + b - label[i])**2
        grad_w1 += 2 *( pre- label[i]) * x[i]**3
        grad_w2 += 2 *( pre - label[i])*x[i]**2
        grad_w3 += 2 *( pre - label[i])*x[i]
        grad_b += 2 *( pre- label[i])
    grad_w1 /=  len(x)
    grad_w2 /=  len(x)
    grad_w3 /= len(x)
    grad_b /=  len(x)
    w1 -= grad_w1 * lr
    w2 -= grad_w2 * lr
    w3 -= grad_w3 * lr
    b -= grad_b * lr

print(w1,w2,w3,b)