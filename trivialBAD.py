import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader

def get_data(path,sort_by_len=False,num=None):
    all_text = []
    all_label = []
    with open(path,"r",encoding="utf8") as f:
        all_data = f.read().split("\n")
        if sort_by_len == True:
            all_data = sorted(all_data,key=lambda x:len(x))
    for data in all_data:
        try:
            if len(data) == 0:
                continue
            data_s = data.split("	")
            if len(data_s) != 2:
                continue
            text,label = data_s
            label = int(label)

        except Exception as e:
            print(e)
        else:
            all_text.append(text)
            all_label.append(int(label))
    if num is None:
            return all_text,all_label
    else:
        return all_text[:num], all_label[:num]

def build_word2index(train_text):
    word_2_index =  {"PAD":0,"UNK":1}
    for text in train_text:
        for word in text:
            if word not in word_2_index:
                word_2_index[word] = len(word_2_index)
    return word_2_index


class TextDataset(Dataset):
    def __init__(self,all_text,all_lable):
        self.all_text = all_text
        self.all_lable = all_lable

    def __getitem__(self, index):
        global word_2_index
        text = self.all_text[index]
        text_index = [word_2_index[i] for i in text]
        label = self.all_lable[index]
        text_len = len(text)
        return text_index,label,text_len


    def process_batch_batch(self, data):
        global word_2_index
        batch_text = []
        batch_label = []
        batch_len = []

        for d in data:
            batch_text.append(d[0])
            batch_label.append(d[1])
            batch_len.append(d[2])
        batch_max_len = max(batch_len)

        # batch_text = [i[:max_len] for i in batch_text]


        batch_text = [i + [0]*(batch_max_len-len(i)) for i in batch_text]
        # batch_onehot = []
        # for text_idx in batch_text:
        #     text_onehot = []
        #     for idx in text_idx:
        #         text_onehot.append(index_2_onehot(idx,len(word_2_index)))
        #     batch_onehot.append(text_onehot)
        return torch.tensor(batch_text),torch.tensor(batch_label)


    def __len__(self):
        return len(self.all_text)


class RNN_Model(nn.Module):
    def __init__(self,embedding_num,hidden_num):
        super().__init__()
        # self.embedding = nn.Embedding(corpus_len,embedding_num)
        self.hidden_num = hidden_num
        self.W = nn.Linear(embedding_num,hidden_num)
        self.V = nn.Linear(embedding_num,hidden_num)
        self.U = nn.Linear(embedding_num,hidden_num)
        self.tanh = nn.Tanh()

    def forward(self,x):

        his = torch.zeros(size = (1,self.hidden_num))
        O_all = []

        O = torch.zeros(x.shape[0],x.shape[1],self.hidden_num,device=x.device)
        t = torch.zeros(size=(x.shape[0], self.hidden_num),device=x.device)

        for i in range(x.shape[1]):
            w_emb = x[:,i]

            h = self.W(w_emb)
            h_ = h*0.2 + t*0.8
            h__ = self.tanh(h_)

            t = self.U(h__)
            O[:,i] = t


        return O,t




class Model(nn.Module):
    def __init__(self,corpus_len,embedding_num,hidden_num,class_num):
        super().__init__()
        self.embedding = nn.Embedding(corpus_len,embedding_num)
        self.rnn = RNN_Model(embedding_num,hidden_num)
        self.classifier = nn.Linear(hidden_num,class_num)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,x,label=None):  # batch * sent_len
        # x 是一个batch的句子的编号表示 一个batch中所有句子的长度必须是相同的(预处理过)
        # x [[1,2,3,4], [2,1,6,5], [7,8,9,5],[5,1,4,7]
        # x [batchSize sentLength]
        # x_emb [batchSize,sentLen,embedLen]
        x_emb = self.embedding(x)
        # o [batchSize sentLen embedLen] o保留了每个句子中每个字的embedding
        # t [batchSize embedLen] t保留了每个句子的embedding
        o,t = self.rnn(x_emb)  # t : batch * 1 * hidden_num    o: batch * sent_len * hidden_num

        # O[:,i] = t t是O的一部分


        # t == o[:,-1,:]  == True
        pre = self.classifier(o)

        if label is not None:
            loss = self.loss_fun(pre,label)
            return  loss
        else:
            return torch.argmax(pre,dim=-1)



if __name__=="__main__":
    train_text, train_lable = get_data(os.path.join("/data2/yuanshou/tmp/handai/textData", "train.txt"), True,10000)
    dev_text, dev_lable = get_data(os.path.join("/data2/yuanshou/tmp/handai/textData", "dev.txt"), True,2000)

    word_2_index = build_word2index(train_text + dev_text)

    train_batch_size = 10
    embedding_num = 100
    hidden_num = 100
    epoch = 10
    lr = 0.001
    class_num = len(set(train_lable))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = TextDataset(train_text, train_lable)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False,collate_fn=train_dataset.process_batch_batch)

    dev_dataset = TextDataset(dev_text, dev_lable)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False, collate_fn=dev_dataset.process_batch_batch)

    model = Model(len(word_2_index),embedding_num,hidden_num,class_num).to(device)
    opt = torch.optim.Adam(model.parameters(), lr)

    for e in range(epoch):
        print("*" * 100)

        model.train()
        for bi, (batch_text, batch_label) in tqdm(enumerate(train_dataloader, start=1)):
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)

            loss = model.forward(batch_text,batch_label)
            loss.backward()

            opt.step()
            opt.zero_grad()

            if bi % 50 == 0:
                print(f"loss:{loss:.2f}")

        model.eval()
        right_num = 0
        for bi, (batch_text, batch_label) in tqdm(enumerate(dev_dataloader, start=1)):
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            pre = model.forward(batch_text)

            right_num+=int(torch.sum(pre==batch_label))

        acc = right_num/len(dev_dataset)
        print(f"acc:{acc*100:.3f}%")







