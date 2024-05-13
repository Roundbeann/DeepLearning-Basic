from tqdm import trange

epoch = 1000
lr = 0.001

label = 5

x = 1
for i in trange(10000):
    pre = (4*x - 3)**2
    loss = (pre - label)**2
    # loss = ((4*x - 3)**2 - label)**2
    grad = 2*((4*x - 3)**2 - label)* 2 * (4*x-3) * 4
    x = x - grad * lr

print(x,(4*x - 3)**2,label)