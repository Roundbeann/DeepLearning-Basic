import torch
import torch.nn as nn
import random
from torch.utils.data import  Dataset,DataLoader

def read_data(file,num=None):
    with open(file,"r",encoding="utf-8") as f:
        all_data = f.read().split("\n")
    if num:
        all_data = all_data[:num]
    return all_data

def get_word_2_index(all_text):
    word_2_index = {"PAD":0,"UNK":1,"STA":2,"END":3}
    for text in all_text:
        for w in text:
            word_2_index[w] = word_2_index.get(w,len(word_2_index))

    index_2_word = list(word_2_index)

    return word_2_index,index_2_word


class PDataset(Dataset):
    def __init__(self,all_data):
        self.all_data = all_data


    def __getitem__(self, index):
        text = self.all_data[index]

        # input_text = text[:-1] # B 床前明月光,疑是地上霜 .
        # label_text = text[1:]  # 床前明月光,疑是地上霜. E

        input_idx = [2] + [word_2_index.get(i,1) for i in text]
        label_idx = [word_2_index.get(i,1) for i in text] + [3]

        return torch.tensor(input_idx),torch.tensor(label_idx)

    def __len__(self):
        return len(self.all_data)


class PModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb = nn.Embedding(word_size,embedding_num)
        self.rnn = nn.RNN(embedding_num,rnn_hidden_num,batch_first=True)
        self.lstm = nn.LSTM(embedding_num,rnn_hidden_num)
        self.dropout = nn.Dropout(0.2)
        self.cls = nn.Linear(rnn_hidden_num,word_size)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,x,lable=None,h=None):
        batch_,seql_ = x.shape
        emb = self.emb(x)
        rnn_out1,rnn_out2 = self.rnn.forward(emb,h)
        rnn_out1 = self.dropout(rnn_out1)
        pre = self.cls.forward(rnn_out1)

        if lable is not None:
            loss = self.loss_fun(pre.reshape(batch_*seql_,-1),lable.reshape(-1))
            return loss
        return torch.argmax(pre,dim=-1),rnn_out1

def auto_generate():
    global model
    # model.eval()
    result = ""
    # not_first_idx = ["PAD","UNK",'，','。',""]
    #
    # while True:
    #     word = random.choice(index_2_word)
    #     if word not in not_first_idx:
    #         break
    # result += word
    word = "STA"
    word_index = word_2_index[word]
    h = None

    while True:
        word_index = torch.tensor([[word_index]])
        word_index,h = model.forward(word_index,h=h)
        # h = torch.squeeze(h,dim=0)
        word_index = int(word_index)

        if word_index == 3 or len(result) > 50 :
            break

        result += index_2_word[word_index]
    return result


def acrostic_poem(text):
    text = text[:4]
    p = ['，','。','，','。']
    assert len(text) >= 4

    result = ""  # 床前明月光,疑是地上霜.

    for i in range(4):
        result += text[i]

        word_index = word_2_index.get(text[i],1)
        h = None
        for j in range(4):
            word_index = torch.tensor([[word_index]])
            word_index, h = model.forward(word_index, h=h)
            word_index = int(word_index)
            result += index_2_word[word_index]

        result += p[i]

    return result


if __name__ == "__main__":

    train_data = read_data("../data/古诗生成/poetry_5.txt",200)
    word_2_index, index_2_word = get_word_2_index(train_data)

    batch_size = 10
    epoch = 100
    embedding_num = 100
    rnn_hidden_num = 50
    lr = 0.004
    word_size = len(word_2_index)

    train_dataset = PDataset(train_data)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    model = PModel()
    opt = torch.optim.Adam(model.parameters(),lr=lr)

    for e in range(epoch):
        model.train()
        for text_idx,label_idx in train_dataloader:
            loss = model.forward(text_idx,label_idx)

            loss.backward()
            opt.step()
            opt.zero_grad()

        # print(f"loss:{loss:.3f}")

        sample = auto_generate()
        print(f"{sample} {loss:.3f} {e}")

    while True:
        text = input("请输入:")
        result = acrostic_poem(text)
        print(result)
