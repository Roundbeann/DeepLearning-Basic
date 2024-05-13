import os
import numpy
import torch
import numpy as np
from torch.nn.modules.module import T
from tqdm import tqdm
import torch.nn as nn
from tqdm import trange
from torch.utils.data import Dataset,DataLoader
def read_data(fileName,num = None):
    with open(os.path.join("/data2/yuanshou/tmp/handai/textData",fileName),encoding='utf-8') as f:
        all_data = f.read().split("\n")[:-1]
        all_sentence = [data.split('\t')[0] for data in all_data]
        all_label = [int(data.split('\t')[1]) for data in all_data]
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
            word2index[word] = word2index.get(word,len(word2index))
    wordsize = len(word2index)
    index2embedding = numpy.random.normal(size=(wordsize,embedding_num)).astype(np.float32)
    return word2index,index2embedding
class TextDataset(Dataset):
    def __init__(self,texts,labels,word2index,max_len):
        self.all_text = texts
        self.all_label = labels
        self.word2index = word2index
        self.max_len = max_len
    def __getitem__(self, index):
        text = self.all_text[index][:max_len]
        lebel = self.all_label[index]
        text_2_index = [word2index.get(i,1) for i in text]
        text_2_index += [0]*(max_len-len(text_2_index))
        return torch.tensor(text_2_index),torch.tensor(lebel)
    def __len__(self):
        return len(self.all_label)

class MyRNN(nn.Module):

    def __init__(self,dict_size,hidden_num):
        super().__init__()
        self.input_size = dict_size
        self.hidden_size = hidden_num

        self.embedding = nn.Embedding(dict_size,embedding_num,device=device)
        self.W = nn.Linear(self.input_size,self.hidden_size,device=device)    # 作用于原始输入
        self.U = nn.Linear(self.hidden_size,self.hidden_size,device=device)    # 作用于上一层的输出
        self.tanh = nn.Tanh().to(device)




    def forward(self,emben_x,label = None):
         # = self.embedding(x)

        batch_size,sent_len,embed_num = emben_x.shape
        t = torch.zeros((batch_size, self.hidden_size), device=device)
        result1 = torch.zeros(size=(batch_size,sent_len,self.hidden_size))

        # [ sent_len (batchSize hidden_num)]
        for i in range(sent_len):
            # i = 0 则取到每个batch的第一个字
            # i = 1 则取到每个batch的第二个字
            # i = 2 则取到每个batch的第三个字
            word_i_embedding = emben_x[:,i]
            h1 = self.W(word_i_embedding)
            h2 = h1 + t
            h3 = self.tanh(h2)

            t = self.U(h3)
            result1[:,i] = t
        # result1 【5000 30 128】
        # 30 是 sent_len
        # 上面的循环是遍历sent_len进行的
        # result1 蕴含了所有句子中全部汉字的特征
        # t 蕴含了每个句子的特征 同时也是一句话的最后一个字的特征
        # 因此在RNN中，一个句子最后一个字的特征可以表示整个句子
        return result1,t


    def __call__(self, *args):
        return self.forward(*args)

class MyLSTM(nn.Module):
    def __init__(self,embedding_len,hidden_num):
        super().__init__()
        self.embed_len = embedding_len
        self.hidden_num = hidden_num

        self.F = nn.Linear(embedding_len+hidden_num,hidden_num)
        self.I = nn.Linear(embedding_len+hidden_num,hidden_num)
        self.C = nn.Linear(embedding_len+hidden_num,hidden_num)
        self.O = nn.Linear(embedding_len+hidden_num,hidden_num)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self,x,label = None):
        batchSize,sent_len,embed_len = x.shape
        a_prev = torch.zeros((batchSize,self.hidden_num),device=device)
        c_prev = torch.zeros((batchSize,self.hidden_num),device=device)

        result = torch.zeros((batchSize,sent_len,hidden_num),device=device)

        # 对于一句话中的第 i 个字
        for word_index in range(sent_len):
            word_emb = x[:,word_index]
            word_a_emb = torch.cat([word_emb,a_prev],dim=-1)
            f = self.F(word_a_emb)
            i = self.I(word_a_emb)
            c = self.C(word_a_emb)
            o = self.O(word_a_emb)

            ft = self.sigmoid(f)
            it = self.sigmoid(i)
            cct = self.tanh(c)
            ot = self.sigmoid(o)

            c_next = ft * c_prev + it * cct
            a_next = ot * self.tanh(c_next)

            result[:,word_index] = a_next

            a_prev = a_next
            c_prev = c_next
        return result,(a_prev,c_prev)
        # return result,(c_prev,a_prev)

class LSTMTextCls(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(dict_size,embedding_num)
        self.lstm = MyLSTM(embedding_num,hidden_num)  # 暂且把 lstm 当成是 Linear 看作是一个100 * 200的权重矩阵
        self.classifier = nn.Linear(hidden_num*max_len,cls_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x,label = None):
        # 这里接收的 x 可能是 2 维的 即 x (batchSize * 30)
        # 把 x 的每个元素进行embedding 编码，得到 (batchSize * 30 * embed_len)
        # x: (batchSize * 30 * embed_len) @ (embed_len * hidden_num) = (batchSize * 30 * hidden_num)
        embed_x = self.embedding(x)
        # output1 [100 30 128] = 对应的就是 RNN 模型的第一个输出 (batchSize * 30 * hidden_num)
        # 这里的output1 保留了中间维度 30，记录了100个句子中每个字的特征
        # output2 [1 100 128] = 对应的是 RNN 模型输出的余项 （ 1 * tatchSize * hidden_num)
        # 这里的output2 仅记录了100个句子每个句子的特征
        lstm_output1, lstm_output2 = self.lstm(embed_x)

        # feature_0 = output1[:, 0]
        feature = lstm_output1.reshape(lstm_output1.shape[0],-1).to(device)
        # feature_0     [100 128]
        # classifier    [128 10 ]
        # pre           [100 10 ]
        pre = self.classifier(feature)

        if label is not None:
            loss = self.loss(pre,label)
            return loss
        if label is None:
            self.predict = torch.argmax(pre, dim=-1).detach().to("cpu").numpy().tolist()
            return self.predict

def test_file():
    global model,device,word2index,index2embedding

    test_text,test_label = read_data("test.txt")
    assert len(test_text) == len(test_label)
    test_onehot_dataset = TextDataset(test_text, test_label, word2index, max_len)
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
    hidden_num = 128
    cls_num = len(set(train_label))
    batch_size = 300
    lr = 0.002
    epoch = 20
    embedding_num = 500
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # W1 【3071 30】  【word_count_in_dict,max_len】
    # W2 【900 10】     【max_len * hidden_num,cls_num】
    word2index,index2embedding = build_curpus(train_text)
    embed_len = 255
    dict_size = len(word2index)


    model = LSTMTextCls().to(device)
    optim = torch.optim.AdamW(model.parameters(),lr = lr)

    train_dataset = TextDataset(train_text,train_label,word2index,max_len)
    # a = trainOnehotDataset[0]
    train_dataloader = DataLoader(train_dataset,batch_size)

    dev_dataset = TextDataset(dev_text,dev_label,word2index,max_len)
    # a = trainOnehotDataset[0]
    dev_dataloader = DataLoader(dev_dataset,1000)


    for e in trange(epoch):
        model.train()
        for texts,labels in tqdm(train_dataloader):
            texts = texts.to(device)
            labels = labels.to(device)
            loss = model(texts,labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
        print(f"\nloss{loss:.2f}")
        result_dev = []
        model.eval()
        for texts, labels in dev_dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            pre = model(texts)
            # pre = model.predict
            result_dev.extend(pre)

        dev_acc = sum([int(int(i) == j) for i, j in zip(result_dev, dev_label)]) / len(dev_label)
        print(f"dev_acc{dev_acc}")



    print(model.state_dict())

    test_file()