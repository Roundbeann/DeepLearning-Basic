import os
import numpy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from tqdm import trange
from torch.utils.data import Dataset,DataLoader
def read_data(fileName,num = None):
    with open(os.path.join("/data2/yuanshou/tmp/handai/textData",fileName),encoding='utf-8') as f:
        all_data = f.read().split("\n")[:-1]
        all_sentence = [data.split('\t')[0] for data in all_data]
        all_label = [data.split('\t')[1] for data in all_data]
        if(num is not None):
            all_sentence = all_sentence[:num]
            all_label = all_label[:num]
        return all_sentence,all_label
# 建立语料库
def build_curpus(train_text):
    word2index = {"<PAD>":0,"<UNK>":1}
    init_index = 2
    for text in train_text:
        for word in text:
            # 简洁写法：
            word2index[word] = word2index.get(word,len(word2index))
            # 冗余写法：
            # if word in word2index.keys():
            #     continue
            # else:
            #     word2index[word] = init_index
            #     init_index = init_index + 1
    wordsize = len(word2index)
    # index2embedding = numpy.random.normal(0,1,size=(wordsize,embedding_num)).astype(np.float32)
    index2embedding = nn.Embedding(wordsize,embedding_num,dtype=torch.float64)
    return word2index,index2embedding
class OHDataset(Dataset):
    def __init__(self,texts,labels,word2index,index2embedding,max_len):
        self.texts = texts
        self.labels = labels
        self.word2index = word2index
        self.index2embedding = index2embedding
        self.max_len = max_len
    def __getitem__(self, index):
        # 1.根据index 获取第index条数据
        # 2.裁剪/填充数据到max_len长度
        # 3.将中文文本转换为onehot编码
        text = self.texts[index]
        label = int(self.labels[index])
        text2onehot = []
        if len(text)>=self.max_len:
            text = text[:self.max_len]
            for i in text:
                try:
                    text2onehot.append(self.index2embedding(torch.tensor(self.word2index[i])).tolist())
                except:
                    text2onehot.append(self.index2embedding(torch.tensor(self.word2index['<UNK>'])).tolist())
            text2onehot = np.array(text2onehot).astype(np.float32)
        elif len(text)<self.max_len:
            pad_num = self.max_len - len(text)
            for i in text:
                try:
                    text2onehot.append(self.index2embedding(torch.tensor(self.word2index[i])).tolist())
                except:
                    text2onehot.append(self.index2embedding(torch.tensor(self.word2index['<UNK>'])).tolist())
            for i in range(pad_num):
                text2onehot.append(self.index2embedding(torch.tensor(self.word2index['<PAD>'])).tolist())
            text2onehot = np.array(text2onehot).astype(np.float32)
        return text2onehot,label
    def __len__(self):
        return len(self.labels)

class OHModel(nn.Module):
    def __init__(self,max_len,embedding_num,hidden_num,cls_num):
        super().__init__()
        self.linear1 = nn.Linear(embedding_num,hidden_num)
        self.active = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear2 = nn.Linear(max_len * hidden_num,cls_num)
        self.cross_loss = nn.CrossEntropyLoss()
    def forward(self,text2embedding,labels = None):
        hidden = self.linear1(text2embedding)
        sig_hidden = self.active(hidden)
        sig_hidden_flatten = self.flatten(sig_hidden)
        p = self.linear2(sig_hidden_flatten)
        self.predict = torch.argmax(p,dim = 1).detach().to('cpu').numpy().tolist()
        # 注意这里p的形状是 【5 3】 是 softmax之前的结果
        # [87, 54, 68],
        # [52, -60, 56],
        # [77, 36, 578],
        # [-85, 25, 34],
        # [-44, 85, 54],
        # labels的形状是 【5】
        # [0, 1, 0, 2, 1]
        if labels is not None:
            # CrossEntropyLoss 计算了
            # 1.softmax()
            # 2.argmax()
            # 3.loss()
            loss = self.cross_loss(p,labels)
            return loss

def test_file():
    global model,device,word2index,index2embedding

    test_text,test_label = read_data("test.txt")
    assert len(test_text) == len(test_label)
    test_onehot_dataset = OHDataset(test_text, test_label, word2index, index2embedding, max_len)
    # a = trainOnehotDataset[0]
    test_dataloader = DataLoader(test_onehot_dataset, 2000)

    result = []
    for text,label in test_dataloader:
        text = text.to(device)
        label = label.to(device)
        model(text)
        pre = model.predict
        result.extend(pre)
        # acc = torch.tensor(pre == label, dtype=torch.float32).mean()
        # print(f"\ntest_acc{acc}")
    acc = sum([int(i == int(j)) for i, j in zip(result,test_label)]) / len(test_label)
    print(acc)


if __name__ == "__main__":
    train_text,train_label = read_data("train.txt",20000)
    dev_text, dev_label = read_data("dev.txt")

    assert len(train_text) == len(train_label)
    assert len(dev_text) == len(dev_label)

    max_len = 30
    hidden_num = 30
    cls_num = len(set(train_label))
    batch_size = 100
    lr = 0.002
    epoch = 20
    embedding_num = 200
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # W1 【3071 30】  【word_count_in_dict,max_len】
    # W2 【900 10】     【max_len * hidden_num,cls_num】
    word2index,index2embedding = build_curpus(train_text)
    # word_count_in_dict = len(word2index)

    model = OHModel(max_len,embedding_num,hidden_num,cls_num).to(device)
    optim = torch.optim.AdamW(model.parameters(),lr = lr)

    train_onehot_dataset = OHDataset(train_text,train_label,word2index,index2embedding,max_len)
    # a = trainOnehotDataset[0]
    train_dataloader = DataLoader(train_onehot_dataset,batch_size)

    dev_onehot_dataset = OHDataset(dev_text,dev_label,word2index,index2embedding,max_len)
    # a = trainOnehotDataset[0]
    dev_dataloader = DataLoader(dev_onehot_dataset,1000)


    for e in trange(epoch):
        for texts,labels in tqdm(train_dataloader):
            texts = texts.to(device)
            labels = labels.to(device)
            loss = model(texts,labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
        print(f"\nloss{loss:.2f}")

        result_dev = []
        for texts,labels in tqdm(dev_dataloader):
            texts = texts.to(device)
            labels = labels.to(device)
            model(texts)
            pre = model.predict
            result_dev.extend(pre)

        dev_acc = sum([int(i == int(j)) for i, j in zip(result_dev, dev_label)]) / len(dev_label)
        print(f"dev_acc{dev_acc}")

    print(model.state_dict())

    test_file()