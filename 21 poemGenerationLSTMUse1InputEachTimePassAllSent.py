import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

from tqdm import trange
from tqdm import tqdm

import random

def get_word2index(all_text):
    word2index = {"UNK":1,"PAD":0}
    for text in all_text:
        for c in text:
            word2index[c] = word2index.get(c,len(word2index))
    index2word = list(word2index)
    return word2index,index2word

def read_poem(path = "/data2/yuanshou/tmp/poemData/poetry_5.txt",num = None):
    with open(path,"r",encoding="utf-8") as f:
        all_data = f.read().split("\n")
    return all_data[:num]

class PoemDataset(Dataset):
    def __init__(self,allData):
        self.allData = allData
        pass

    def __getitem__(self, index):
        text = self.allData[index]
        input_text = text[:-1]
        label_text = text[1:]

        input_idx = [word2index.get(c, 1) for c in input_text]
        label_idx = [word2index.get(c, 1) for c in label_text]

        return torch.tensor(input_idx),torch.tensor(label_idx)

    def __len__(self):
        return len(self.allData)

class PoemGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(dict_len,embedding_len)
        self.lstm = nn.LSTM(embedding_len,hidden_num,batch_first=True)
        self.cls = nn.Linear(hidden_num,dict_len)
        self.loss = nn.CrossEntropyLoss()
    def forward(self,pre_idx,later_idx=None):
        batch_size ,sent_len = pre_idx.shape
        embed_pre_idx = self.embedding(pre_idx)
        output1,output2 = self.lstm(embed_pre_idx)
        # output1[:, -1] == output2[0]
        # output1 【batchSize sent_len hidden_num】
        # output2 【1 batchSize hidden_num】
        if later_idx is not None:
            pre = self.cls(output1)
            loss = self.loss(pre.reshape(batch_size*sent_len,-1),later_idx.reshape(-1))
            return loss
        else:
            pre = self.cls(output1)
            return torch.argmax(pre,dim=-1).reshape(-1).detach().to("cpu").numpy().tolist()[-1]
            # pre2 = self.cls(output2)
            # return torch.argmax(pre2, dim=-1).reshape(-1).detach().to("cpu").numpy().tolist()[0]
        pass

def generate():
    not_first_c = ["PAD","UNK","，","。"]
    flag = True
    while True:
        result = ""
        word = random.choice(index2word)
        h = None
        if word in not_first_c:
            break

        result += word

        for i in range(23):
            sent_idx = [word2index[c] for c in result]
            word_index = torch.tensor([sent_idx])
            pre = model.forward(word_index)
            result += index2word[pre]
            # word_index = torch.tensor([[pre]])
        flag = False
        return result


if __name__ == "__main__":
    train_data = read_poem()
    word2index,index2word = get_word2index(train_data)

    batch_size = 200
    epoch = 100

    embedding_len = 200
    dict_len = len(index2word)
    hidden_num = 128
    lr = 0.01

    trainDataset = PoemDataset(train_data)
    trainDataLoader = DataLoader(trainDataset,batch_size=batch_size, shuffle = False)

    device = "cuda:0" if torch.cuda.is_available() == True else "cpu"

    model = PoemGeneration()
    optim = torch.optim.AdamW(model.parameters(),lr = lr)
    for e in range(epoch):
        model.train()
        for pre_idx, later_idx in trainDataLoader:
            loss = model.forward(pre_idx,later_idx)
            loss.backward()
            optim.step()
            optim.zero_grad()

        model.eval()
        sample = generate()
        print(str(e)+" "+sample +"  loss:" +str(float(loss)))
    print()
