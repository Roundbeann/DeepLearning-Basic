import numpy as np
import struct # 读取数据
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
            yield data[i:i+batch_size],i,i+batch_size
        else:
            yield data[i:data_num],i,data_num
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
def softmax(scoreList):
    res = []
    for scores in scoreList:
        total = np.exp(scores).sum()
        scores = np.exp(scores) / total
        res.append(scores)
    return np.array(res)

def get_data():
    train_data = load_images("/data2/yuanshou/tmp/handai/mnist/train-images-idx3-ubyte") / 255
    train_label = load_labels("/data2/yuanshou/tmp/handai/mnist/train-labels-idx1-ubyte")

    test_data = load_images("/data2/yuanshou/tmp/handai/mnist/t10k-images-idx3-ubyte") / 255
    test_label = load_labels("/data2/yuanshou/tmp/handai/mnist/t10k-labels-idx1-ubyte")
    return train_data,train_label,test_data,test_label


class Linear:
    def __init__(self,inputSize,outputSize):
        self.weight = np.random.normal(0, 1, size=(inputSize, outputSize))

    def forward(self,x):
        self.x = x
        return x @ self.weight

    def backward(self,G):
        G_weight = self.x.T @ G
        self.weight = self.weight - lr * G_weight
        G_x = G @ self.weight.T
        return G_x

class Sigmoid:
    def forward(self,x):
        self.r = sigmoid(x)
        return self.r
    def backward(self,G):
        G_x = G * self.r * (1 - self.r)
        return G_x

class Softmax:
    def forward(self, x):
        self.r = softmax(x)
        return self.r
    def backward(self,G):   # G传的是label
        G_x = self.r - G #pre-label
        return G_x



if __name__ == "__main__":
    # /255 数据归一化
    train_data, train_label, test_data, test_label =get_data()
    print("over")
    train_label  = make_one_hot(train_label)
    test_label  = make_one_hot(test_label)
    # train_data    【60000,784】
    # test_data     【10000,784】
    # train_label   【60000,10】
    # test_label    【10000,10】

    # 下面这几个参数都是全局变量
    epochs = 10
    lr = 0.00015
    batch_size = 10000
    hidden_num = 256

    linear1_layer = Linear(784,hidden_num)
    sigmoid_layer = Sigmoid()
    linear2_layer = Linear(hidden_num,10)
    softmax_layer = Softmax()
    # 训练过程
    for epoch in trange(epochs):
        for X,start,end in DataLoader(train_data,batch_size=10000):

            # ------------ forward ------------
            # h = linear1_layer.forward(X)
            # sig_h = sigmoid_layer.forward(h)
            # p = linear2_layer.forward(sig_h)
            # pre = softmax_layer.forward(p)

            x = linear1_layer.forward(X)
            x = sigmoid_layer.forward(x)
            x = linear2_layer.forward(x)
            pre = softmax_layer.forward(x)


            # label 【10000,10】
            label = train_label[start:end]
            # loss 【10000,10】
            loss = label * np.log(pre)
            avg_loss = -loss.mean(axis=0).sum()
            print(avg_loss)

            # G2 = pre - label
            # Grad_sig_h = linear2_layer.backward(G2)
            # Grad_h =sigmoid_layer.backward(Grad_sig_h)
            # linear1_layer.backward(Grad_h)

            G = label
            G = softmax_layer.backward(G)
            G = linear2_layer.backward(G)
            G = sigmoid_layer.backward(G)
            G = linear1_layer.backward(G)


            pass

    # 测试过程
    for X, start, end in DataLoader(test_data, batch_size):

        # ------------ forward ------------
        # h = linear1_layer.forward(X)
        # sig_h = sigmoid_layer.forward(h)
        # p = linear2_layer.forward(sig_h)
        # pre = softmax_layer.forward(p)

        x = linear1_layer.forward(X)
        x = sigmoid_layer.forward(x)
        x = linear2_layer.forward(x)
        pre = softmax_layer.forward(x)


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