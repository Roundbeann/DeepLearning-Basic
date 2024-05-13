import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import numpy
from torch import nn
from torch import optim
Chinese = ["粉红色","苹果" ,"香蕉", "橘子", "米", "梨", "蓝色", "黑色", "黄色", "绿色", "红色", "白色"]
English = ["pink","apple" , "banana", "orange", "rice", "peach", "blue", "black", "yellow", "green" ,"red" ,"white"]

def getIndexAndDict(Chinese,English):
    chi2index = {"UNK":1,"PAD":0,"STA":2,"END":3}
    eng2index = {"UNK": 1, "PAD": 0}
    mix2index = {"UNK": 1, "PAD": 0}
    for word in Chinese:
        for i in word:
            chi2index[i] = chi2index.get(i,len(chi2index))
            mix2index[i] = mix2index.get(i,len(mix2index))

    for alpha in English:
        for i in alpha:
            eng2index[i] = eng2index.get(i,len(eng2index))
            mix2index[i] = mix2index.get(i, len(mix2index))
    index2chi = np.array(list(chi2index))
    index2eng = np.array(list(eng2index))
    index2mix = np.array(list(mix2index))
    return chi2index,eng2index,index2chi,index2eng,mix2index,index2mix

def transToIndex(Chinese,English):
    chiIndex = []
    engIndex = []

    chiIndexMix = []
    engIndexMix = []
    for word in Chinese:
        word = word[:chiMaxLen]
        wordIndex = [2]+[chi2index.get(i,1) for i in word]+[3]+(chiMaxLen - len(word)) * [0]
        wordIndexMix = [mix2index.get(i,1) for i in word]+(chiMaxLen - len(word)) * [0]
        chiIndex.append(wordIndex)
        chiIndexMix.append(wordIndexMix)

    for word in English:
        word = word[:engMaxLen]
        wordIndex = [eng2index.get(i,1) for i in word]+(engMaxLen - len(word)) * [0]
        wordIndexMix = [mix2index.get(i,1) for i in word]+(engMaxLen - len(word)) * [0]
        engIndex.append(wordIndex)
        engIndexMix.append(wordIndexMix)

    engComChi = [i +j for i ,j in zip(engIndexMix,chiIndexMix)]
    chiIndex = torch.tensor(chiIndex)
    engIndex = torch.tensor(engIndex)
    engComChi = torch.tensor(engComChi)
    return chiIndex,engIndex,engComChi

class transDataset(Dataset):
    def __init__(self,engIndex,chiIndex):
        assert len(engIndex) == len(chiIndex)
        self.engIndex = engIndex
        self.chiIndex = chiIndex
    def __getitem__(self, index):
        return self.engIndex[index],self.chiIndex[index]
    def __len__(self):
        return len(self.engIndex)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.LSTM(embedLen,hiddenNum,batch_first=True)

    def forward(self,batchEngEmb):
        _ , output2 = self.model(batchEngEmb)
        return output2

    def __call__(self, *args):
        return self.forward(*args)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.LSTM(embedLen,hiddenNum,batch_first=True)
        self.cls = nn.Linear(hiddenNum,chiDictLen)
    def forward(self,batchChiEmb,engHistory):
        output1, engHistory = self.model(batchChiEmb, engHistory)
        pre = self.cls(output1)
        return pre,engHistory



class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddingEng = nn.Embedding(engDictLen, embedLen)
        self.embeddingChi = nn.Embedding(chiDictLen,embedLen)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.cls = nn.Linear(hiddenNum,chiDictLen)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, batchEng,batchChi=None):
        batchEngEmb = self.embeddingEng(batchEng)
        eachSentFeat = self.encoder(batchEngEmb)
        if batchChi is not None:
            batchChiEmb = self.embeddingChi(batchChi[:, :-1])
            pre,engHistory =  self.decoder(batchChiEmb,eachSentFeat)
            loss = self.loss(pre.reshape(pre.shape[0]*pre.shape[1],-1),batchChi[:,1:].reshape(-1))
            return loss
        else:
            res = [2]
            for i in range(4):
                batchChi = torch.tensor([[res[-1]]])
                batchChiEmb = self.embeddingChi(batchChi)
                pre,eachSentFeat  = self.decoder(batchChiEmb, eachSentFeat)
                pre = pre.detach().to("cpu")
                nextIndex = int(torch.argmax(pre.reshape(-1)))
                res.append(nextIndex)
            res= np.array(res[1:-1])
            return index2chi[res]
            pass
        # pass
    def backward(self):
        pass
    def __call__(self, *args, **kwargs):
        return self.forward(*args)


if __name__ == "__main__":
    # 得到中文字典、英文字典、中英文混合字典
    chi2index, eng2index, index2chi, index2eng,mix2index,index2mix = getIndexAndDict(Chinese,English)
    # 定义截断长度
    chiMaxLen = 3
    engMaxLen = 7
    # 得到中文语料、英文语料、中英文联合语料的index表示
    # chi2Index【batchsize,chiMaxLen】
    # engMaxLen【batchsize,engMaxLen】
    # engComChi【batchsize,chiMaxLen + engMaxLen】
    chi2Index, eng2Index,engComChi = transToIndex(Chinese,English)

    tranSet = transDataset(eng2Index,chi2Index)
    transLoader = DataLoader(tranSet,batch_size=3)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mixedDictLen = len(index2mix)
    chiDictLen = len(index2chi)
    engDictLen = len(index2eng)

    embedLen = 200
    hiddenNum = 128

    lr = 0.001
    epoch = 200

    model = MyModel()

    opt = optim.Adam(model.parameters(),lr = lr)

    for e in range(epoch):
        model.train()
        for batchEng,batchChi in transLoader:
            loss = model.forward(batchEng,batchChi)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(loss)
        model.eval()
        sample = "banana"
        sampleIndex = torch.tensor([[eng2index.get(i,1) for i in sample]+[0]*(engMaxLen-len(sample))])
        pre = model.forward(sampleIndex)
        print(pre)
        pass

    pass
