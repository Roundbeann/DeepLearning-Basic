import pandas as pd
import os
import jieba
from tqdm import tqdm
from tqdm import trange
import torch
import torch.nn as nn
import numpy as np
import pickle

def getData(path='/data2/yuanshou/tmp/handai/word2vectorData/textData.csv'):
    text = pd.read_csv(path,encoding = "gbk",names = ["text"])
    text = text["text"].tolist()
    stop_words = get_stop_word()
    result = []
    for t in tqdm(text,desc="Cutting the sentences..."):
        tc = jieba.lcut(t)
        tc =  [i for i in tc if i not in stop_words]
        result.append(tc)
    with open('/data2/yuanshou/tmp/handai/word2vectorData/allData.pkl', 'wb')as  f:
        pickle.dump(result, f)



def get_stop_word(path = '/data2/yuanshou/tmp/handai/word2vectorData/stopwords.txt'):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().split("\n")


def build_word(train_text):
    word2index = {}
    for text in train_text:
        word2index["UNK"] = 0
        for word in text:
            word2index[word] = word2index.get(word,len(word2index))
    return word2index


class Word2Vec(nn.Module):
    def __init__(self,word_size,emnedding_len,device):
        super().__init__()
        self.w1 = nn.Linear(word_size, emnedding_len,device=device,bias=False)
        self.w2 = nn.Linear(emnedding_len,word_size,device=device,bias=False)
        self.loss = nn.CrossEntropyLoss().to(device)

    def forward(self,X,label = None):
        h = self.w1(X)
        pass
        p = self.w2(h)
        if label is not None:
            loss = self.loss(p,label)
            return loss


if __name__ == "__main__":
    # all_data = getData()
    # 从文件中读取list
    with open('/data2/yuanshou/tmp/handai/word2vectorData/allData.pkl', 'rb') as f:
        all_data = pickle.load(f)

    word2index = build_word(all_data)

    word_size = len(word2index)
    embedding_num = 100
    n_gram = 2

    batch_size = 100
    lr = 0.006
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Word2Vec(word_size, embedding_num,device=device)

    optim = torch.optim.Adam(model.parameters(),lr = lr)
    epoch = 100

    for e in trange(epoch):
        for text in tqdm(all_data):
            cur_words = []
            other_words = []
            for i,word in enumerate(text):
                cur = np.zeros(word_size)
                cur[word2index[word]] = 1
                oth = text[max(0,i-n_gram):i]+text[i+1:i+n_gram+1]
                oth_label = [word2index.get(i,1) for i in oth]
                cur_onehot = np.tile(cur,(len(oth),1))
                other_words.extend(oth_label)
                cur_words.extend(cur_onehot)
                pass
            cur_words = torch.tensor(cur_words,dtype=torch.float32,device=device)
            other_words = torch.tensor(other_words,dtype=torch.int64,device=device)
            if len(cur_words) ==0:
                continue
            loss = model(cur_words,other_words)
            print(loss)
            loss.backward()
            optim.step()
            optim.zero_grad()
            # model(cur_word)cur_word
        with open(f"/data2/yuanshou/tmp/handai/word2vectorData/modelParam{str(e)}.pkl","wb") as f:
            pickle.dump(model.state_dict(),f)
        pass

