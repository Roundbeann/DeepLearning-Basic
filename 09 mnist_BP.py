import numpy as np
import struct
import matplotlib.pyplot as plt
from tqdm import trange
def load_labels(file):
    with open(file,"rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]),dtype = np.int32)

def load_images(file):
    with open(file,"rb") as f:
        data = f.read()
    magic_number,num_items, rows, cols = struct.unpack(">iiii",data[:16])
    return np.asanyarray(bytearray(data[16:]),dtype = np.uint8).reshape(num_items,-1)

def make_one_hot(labels,class_num = 10):
    res = []
    for i in labels:
        onehot = [0,0,0,0,0,0,0,0,0,0]
        onehot[i] = 1
        res.append(onehot)
    return np.array(res)


def DataLoader(data,batch_size = 300):
    data_num = len(data)
    start = 0
    for i in range(0,data_num,batch_size):
        if i+batch_size<=data_num:
            yield data[i:i+batch_size],batch_size,i,i+batch_size
        else:
            yield data[i:data_num],data_num-i,i,data_num

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(scoreList):
    res = []
    for scores in scoreList:
        total = np.exp(scores).sum()
        scores = np.exp(scores) / total
        res.append(scores)
    return np.array(res)

if __name__ == "__main__":
    # /255 数据归一化
    train_data = load_images("/data2/yuanshou/tmp/handai/mnist/train-images-idx3-ubyte") / 255
    train_label = load_labels("/data2/yuanshou/tmp/handai/mnist/train-labels-idx1-ubyte")

    test_data = load_images("/data2/yuanshou/tmp/handai/mnist/t10k-images-idx3-ubyte") / 255
    test_label = load_labels("/data2/yuanshou/tmp/handai/mnist/t10k-labels-idx1-ubyte")
    print("over")
    train_label  = make_one_hot(train_label)
    test_label  = make_one_hot(test_label)
    # train_data    【60000,784】
    # test_data     【10000,784】
    # train_label   【60000,10】
    # test_label    【10000,10】
    epochs = 200
    lr = 0.00015
    hidden_num = 256
    W1 = np.random.normal(0,1,size=(784,hidden_num))
    W2 = np.random.normal(0,1,size=(hidden_num,10))

    # 训练过程
    for epoch in trange(epochs):
        for X, batch_size,start,end in DataLoader(train_data,batch_size=10000):
            h = X @ W1
            sig_h = sigmoid(h)
            p = sig_h @ W2
            pre = softmax(p)
            # label 【10000,10】
            label = train_label[start:end]
            # loss 【10000,10】
            loss = label * np.log(pre)
            avg_loss = -loss.mean(axis=0).sum()
            print(avg_loss)
            # 根据链式法则，求loss 对 W2的导数，需要求loss对W2的函数的导数
            # 也就是需要先求loss对 p 和 pre 的导数

            # 求loss对p的导数 即求loss对交叉熵和softmax的联合导数
            # 结论：loss'(p) = pre - label
            # G2【10000,10】
            G2 = pre - label

            # 更新 W2,需要求loss对W2的导数
            # sig_h.T 【256,10000】 G2【10000,10】
            # Grad_W2 【256,10000】 和 W2 的形状相同
            Grad_W2 = sig_h.T @ G2

            # 根据链式法则，求loss 对 W1的导数，需要求loss对W1的函数的导数
            # 也就是需要先求loss对 h sig_h p pre 的导数

            # G2【10000,10】 W2.T【10,256】
            # Grad_sig_h【10000,256】
            Grad_sig_h = G2 @ W2.T

            # Grad_sig_h 【10000,256】 sig_h【10000,256】 1-sig_h【10000,256】
            # Grad_h = Grad_sig_h * sigmoid(h) * (1-sigmoid(h))
            Grad_h = Grad_sig_h * sig_h * (1-sig_h)
            G1 = Grad_h

            # X.T 【784,10000】 G1【10000,256】
            # Grad_W1【784,256】 和 W1 的形状相同
            Grad_W1 = X.T @ G1

            W1 = W1 - lr * Grad_W1
            W2 = W2 - lr * Grad_W2

            pass

    # 测试过程
    for X, batch_size, start, end in DataLoader(test_data, batch_size=2000):
        h = X @ W1
        sig_h = sigmoid(h)
        p = sig_h @ W2
        pre = softmax(p)
        # label 【batch_size,10】
        label = test_label[start:end]
        # loss 【batch_size,10】
        loss = label * np.log(pre)
        avg_loss = -loss.mean(axis=0).sum()
        print(avg_loss)
        print([i.argmax() for i in label[:20]])
        print([i.argmax() for i in pre[:20]])
        accuracy = (np.array([i.argmax() for i in label])==np.array([i.argmax() for i in pre])).astype(int).mean()
        print(f"accuracy{accuracy}")


    pass