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
    def __call__(self, x):
        return self.forward(x)
class Sigmoid:
    def forward(self,x):
        self.r = sigmoid(x)
        return self.r
    def backward(self,G):
        G_x = G * self.r * (1 - self.r)
        return G_x
    def __call__(self, x):
        return self.forward(x)

class Softmax:
    def forward(self, x):
        self.r = softmax(x)
        return self.r
    def backward(self,G):   # G传的是label
        G_x = self.r - G #pre-label
        return G_x
    def __call__(self, x):
        return self.forward(x)

class MyModel:
    def __init__(self,layers):
        self.layers = layers

    def forward(self,x, label = None):
        for layer in self.layers:
            x = layer(x)
        self.x = x
        if label is not None:
            self.label = label
            loss = - label * np.log(x)
            avg_loss = loss.mean(axis=0).sum()
            return avg_loss

    def backward(self):
        G = self.label
        for layer in self.layers[::-1]:
            G = layer.backward(G)
    def __call__(self, *args):  #使用变参
        return self.forward(*args)

if __name__ == "__main__":

    train_data, train_label, test_data, test_label =get_data()
    print("over")
    train_label  = make_one_hot(train_label)
    test_label  = make_one_hot(test_label)

    epochs = 10
    lr = 0.00015
    batch_size = 10000
    hidden_num = 256


    model = MyModel([
        Linear(784, hidden_num),
        Sigmoid(),
        Linear(hidden_num, 10),
        Softmax()
    ])


    # 训练过程
    for epoch in trange(epochs):
        for x,start,end in DataLoader(train_data,batch_size=10000):

            label = train_label[start:end]

            loss = model(x,label)

            print(loss)

            model.backward()

            pass


    for x, start, end in DataLoader(test_data, batch_size):

        model(x)

        label = test_label[start:end]
        # model.x 其实就是最后的 predict
        loss = -label * np.log(model.x)
        avg_loss = loss.mean(axis=0).sum()

        print(f"avg_loss:{avg_loss}")
        print([i.argmax() for i in label[:20]])
        print([i.argmax() for i in model.x[:20]])

        accuracy = (np.array([i.argmax() for i in label]) == np.array([i.argmax() for i in model.x])).astype(int).mean()
        print(f"accuracy:{accuracy}")

    pass