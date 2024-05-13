import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import numpy
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

Chinese = ["猫","米","香蕉","粉红色","苹果" , "橘子",  "梨", "蓝色", "黑色", "黄色", "绿色", "红色", "白色"]
English = ["cat","rice","banana","pink","apple" ,  "orange",  "peach", "blue", "black", "yellow", "green" ,"red" ,"white"]

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
        # wordIndex = [2]+[chi2index.get(i,1) for i in word]+[3]+(chiMaxLen - len(word)) * [0]
        # wordIndexMix = [mix2index.get(i,1) for i in word]+(chiMaxLen - len(word)) * [0]
        wordIndex = [chi2index.get(i,1) for i in word]
        wordIndexMix = [mix2index.get(i,1) for i in word]
        chiIndex.append(wordIndex)
        chiIndexMix.append(wordIndexMix)

    for word in English:
        word = word[:engMaxLen]
        # wordIndex = [eng2index.get(i,1) for i in word]+(engMaxLen - len(word)) * [0]
        # wordIndexMix = [mix2index.get(i,1) for i in word]+(engMaxLen - len(word)) * [0]
        wordIndex = [eng2index.get(i,1) for i in word]
        wordIndexMix = [mix2index.get(i,1) for i in word]
        engIndex.append(wordIndex)
        engIndexMix.append(wordIndexMix)

    engComChi = [i +j for i ,j in zip(engIndexMix,chiIndexMix)]
    # chiIndex = torch.tensor(chiIndex)
    # engIndex = torch.tensor(engIndex)
    # engComChi = torch.tensor(engComChi)
    return chiIndex,engIndex,engComChi

class transDataset(Dataset):
    def __init__(self,engIndex,chiIndex):
        assert len(engIndex) == len(chiIndex)
        self.engIndex = engIndex
        self.chiIndex = chiIndex

    def __getitem__(self, index):
        engIndex = torch.tensor(self.engIndex[index][:engMaxLen] + (engMaxLen-len(self.engIndex[index]))*[0])
        chiIndex = torch.tensor([2]+self.chiIndex[index][:chiMaxLen] +[3]+ (chiMaxLen-len(self.chiIndex[index]))*[0])
        return engIndex,chiIndex,len(self.engIndex[index]), len(self.chiIndex[index])+2
    def __len__(self):
        return len(self.engIndex)
    # 思考：
    # 原来是把一个batch的英文单词连带padding的embedding传给rnn
    # embedding当中的padding记录的历史信息是无效的
    # 因此对rnn返回的对英文单词的编码，只截取到每个英文单词的有效位
    # 英文单词的有效位由eng_len记录
    # 1.【手动截取】有用历史信息：
    # enr_out, hid_ = self.encoder.forward(eng_e)
    # enr_out_new = torch.zeros_like(enr_out, device=eng_index.device)
    # hid_new = torch.zeros_like(hid_)
    # for i in range(len(eng_len)):
    #     l = eng_len[i]
    #     enr_out_new[i][:l] = enr_out[i, :l]
    #     hid_new[:, i] = enr_out[i][l - 1:l]
    # 2.【自动截取】有用历史信息：
    # pack = pack_padded_sequence(eng_e, eng_len, batch_first=True, enforce_sorted=False)
    # encoder_out, hidden = self.encoder.forward(pack)

    # 注意，在这里pack_padded_sequence以及和pytorch的RNN搭配好了
    # pack_padded_sequence输出的结果可以直接交给RNN进行处理

    # 不论哪一种方法都需要知道每个单词除padding外的有效长度是多少
    # 这个信息在Dataset当中就要被返回回来

    # 手动实现需要对padding保留的历史信息删除
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.GRU(embedLen,hiddenNum,batch_first=True)

    def forward(self,batchEngEmb,engLen):
        pack = pack_padded_sequence(batchEngEmb,engLen,batch_first=True,enforce_sorted=False)
        output1 , output2 = self.model(pack)
        output1_1,output1_2  = pad_packed_sequence(output1,batch_first=True)
        # 效果：output1_1[0][-1]全为0
        return output2

    def __call__(self, *args):
        return self.forward(*args)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.GRU(embedLen,hiddenNum,batch_first=True)
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
    def forward(self, batchEng,engLen,batchChi=None):
        batchEngEmb = self.embeddingEng(batchEng)
        eachSentFeat = self.encoder(batchEngEmb,engLen)
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
        for  batchEng,batchChi,engLen,chiLen in transLoader:
            loss = model.forward(batchEng,engLen,batchChi)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(loss)
        model.eval()
        sample = "banana"
        sampleIndex = torch.tensor([[eng2index.get(i,1) for i in sample]+[0]*(engMaxLen-len(sample))])
        pre = model.forward(sampleIndex,[len(sample)])
        print(pre)
        pass

    pass
