from torch.utils.data import DataLoader
import argparse
from torch import nn, optim
import numpy as np
import torch
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import math

sequence_length=1
batch_size=450
max_epochs=1
write_perplexities=True


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.sequence_length=sequence_length
        self.batch_size=batch_size
        self.embedding_dim = 128
        self.num_layers = 3
        self.max_epochs=max_epochs
        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(num_embeddings=n_vocab,embedding_dim=self.embedding_dim)
        self.lstm_size = 128
        self.fc = nn.Linear(self.lstm_size, n_vocab)
        self.lstm = nn.LSTM(input_size=self.lstm_size,hidden_size=self.lstm_size,num_layers=self.num_layers,dropout=0.2)

    def forward(self, x, prev_state):
        embed = self.embedding(x) #vector of word
        output, state = self.lstm(embed, prev_state) #pre_state is output of previous state
        # print("OUTTTTTTTTTTTT", output)
        # print(output.size())
        logits = self.fc(output)
        # print(logits.size())
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

class Corpus(torch.utils.data.Dataset):
    def __init__(self):
        self.sequence_length=sequence_length
        self.batch_size=batch_size
        self.max_epochs=max_epochs
        self.words, self.validation_words, self.testing_words = self.load_words()
        word_counts = Counter(self.words)
        self.uniq_words = sorted(word_counts, key=word_counts.get, reverse=True)

        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}


    def load_words(self):
        with open("./train.txt", "r") as f1:
            dat=f1.read()
            dat=dat.replace("\n", " ")
            dat=dat.replace("@", "")
            dat=dat.replace("#", "")
            dat=dat.replace("*", "")
            dat=dat.replace("+", "")
            dat=dat.replace("^", "")
            dat=dat.replace("&", "")
            dat=dat.replace("~", "")
            dat=dat.replace("  ", " ")
            dat=dat.replace("{", "")
            dat=dat.replace("}", "")
            dat=dat.replace("[", "")
            dat=dat.replace("]", "")
            dat=dat.replace("(", "")
            dat=dat.replace(")", "")
            dat=dat.replace(":", "")
            dat=dat.replace("\\", "")
            dat=dat.replace("`", "")
            # dat=dat.replace('"', "")


            sentences=sent_tokenize(dat)
            training_arr=[]
            for line in sentences:
              a=line.strip()
              a=a.replace('"',"")
              training_arr+=["<sent>"] + word_tokenize(a)
        with open("./valid.txt", "r") as f1:
            dat=f1.read()
            dat=dat.replace("\n", " ")
            dat=dat.replace("@", "")
            dat=dat.replace("#", "")
            dat=dat.replace("*", "")
            dat=dat.replace("+", "")
            dat=dat.replace("^", "")
            dat=dat.replace("&", "")
            dat=dat.replace("~", "")
            dat=dat.replace("  ", " ")
            dat=dat.replace("{", "")
            dat=dat.replace("}", "")
            dat=dat.replace("[", "")
            dat=dat.replace("]", "")
            dat=dat.replace("(", "")
            dat=dat.replace(")", "")
            dat=dat.replace(":", "")
            dat=dat.replace("\\", "")
            dat=dat.replace("`", "")
            sentences=sent_tokenize(dat)
            validation_arr=[]
            for line in sentences:
              a=line.strip()
              a=a.replace('"',"")
              validation_arr+=["<sent>"] + word_tokenize(a)
        with open("./test.txt", "r") as f1:
            dat=f1.read()
            dat=dat.replace("\n", " ")
            dat=dat.replace("@", "")
            dat=dat.replace("#", "")
            dat=dat.replace("*", "")
            dat=dat.replace("+", "")
            dat=dat.replace("^", "")
            dat=dat.replace("&", "")
            dat=dat.replace("~", "")
            dat=dat.replace("  ", " ")
            dat=dat.replace("{", "")
            dat=dat.replace("}", "")
            dat=dat.replace("[", "")
            dat=dat.replace("]", "")
            dat=dat.replace("(", "")
            dat=dat.replace(")", "")
            dat=dat.replace(":", "")
            dat=dat.replace("\\", "")
            # dat=dat.replace('"', "")
            dat=dat.replace("`", "")

            sentences=sent_tokenize(dat)
            testing_arr=[]
            for line in sentences:
              a=line.strip()
              a=a.replace('"',"")
              # print("BBBBBBBBBBBBBBBB", (a))
              testing_arr+=["<sent>"] + word_tokenize(a)
            # return data
          # print(return_arr[:10])
        training_arr=self.changeunknownwords(training_arr)
        return training_arr, validation_arr, testing_arr
        # train_df = pd.read_csv('./reddit-cleanjokes.csv')
        # text = train_df['Joke'].str.cat(sep=' ')
        # return text.split(' ')
    def changeunknownwords(self, arr):
        count_dic={}
        for element in arr:
            if element in count_dic:
                count_dic[element]+=1
            else:
                count_dic[element]=1
        for index,element in enumerate(arr):
            if count_dic[element]<=3:
                arr[index]="<unk>"
        return arr

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.sequence_length+1]),
        )

def calculate_perplexity(dataset, model, current_data, savetofile=False, filename=""):
    model.eval()
    sent_count=0
    perplexity=-1
    # print("AAAAAAA",current_data[:10])
    # print(current_data[:10])
    probab_product=0
    word_count=0
    final_answer=0
    sentence_count=0
    state_h, state_c = model.init_state(sequence_length)
    sentence=""

    for i in range(0, len(current_data)-sequence_length):
        # print(current_data[i], word_count)
        if current_data[i]=="<sent>":
            if i!=0:
                # print(word_count)
                perplexity = math.exp((-1/word_count)*probab_product)
                sent_count+=1
                if savetofile:
                    print(perplexity)
                    filename.write(sentence+"\t"+str(perplexity)+"\n")
                final_answer+=perplexity
                sentence=""
                word_count=0
                probab_product=0
        else:
            word_count+=1
            sentence=sentence+" "+current_data[i]

        x_vector=[]
        for w in current_data[i:i+sequence_length]:
            if w in dataset.word_to_index:
                x_vector.append(dataset.word_to_index[w])
            else:
                x_vector.append(dataset.word_to_index["<unk>"])
        # print(x_vector)        
        x = torch.tensor([x_vector])
        # print(x)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        if current_data[i+1] in dataset.word_to_index:
            word_to_predict=dataset.word_to_index[current_data[i+1]]
        else:
            word_to_predict=dataset.word_to_index["<unk>"]
        # print(p[:20])
        # print(len(p))
        # if word_to_predict==-1:
        probab_product+=math.log(p[word_to_predict])
        # else:
        #     probab_product+=math.log(0.001)
        # # print(words)
    perplexity = math.exp((-1/word_count)*probab_product)
    if savetofile:
        print(perplexity)
        filename.write(sentence+"\t"+str(perplexity)+"\n")
    final_answer+=perplexity
    sentence_count+=1
    avg_perplexity=final_answer/(sentence_count)
    if savetofile:
        filename.write(str(avg_perplexity))
    return avg_perplexity
# perplexity=1
# print(dataset.word_to_index)
# print(predict(dataset, model, text='<sent> <sent> <sent> he saw Benny.\n yet'))

def train(dataset, model):
    model.train()
    min_perplexity=1e15
    best_model=-1

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # training_error=[]
    # I am going.
    for epoch in range(max_epochs):
        print("EPOCH: ",epoch)
        state_h, state_c = model.init_state(sequence_length)
        # curr_loss=0

        for batch, (x, y) in enumerate(dataloader):
            # print((x))
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()
            # curr_loss+=loss.item()
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
        # training_error.append(curr_loss)
        torch.save(model, f"./model{epoch}")
        perplexity = calculate_perplexity(dataset=dataset, model=model, current_data=dataset.validation_words, savetofile=False)
        if write_perplexities:
            with open(f"./2019114006-LM{epoch}-train-perplexity", "w") as f1:
                calculate_perplexity(dataset=dataset, model=model, current_data=dataset.words[:30000], savetofile=True,filename=f1)
            with open(f"./2019114006-LM{epoch}-validate-perplexity", "w") as f1:
                perplexity = calculate_perplexity(dataset=dataset, model=model, current_data=dataset.validation_words[:30000], savetofile=True,filename=f1)
            with open(f"./2019114006-LM{epoch}-test-perplexity", "w") as f1:
                calculate_perplexity(dataset=dataset, model=model, current_data=dataset.testing_words[:30000], savetofile=True,filename=f1)
        output_pipe=open("./output.txt", "a")
        output_pipe.write("epoch "+str(epoch)+": "+str(perplexity)+"\n")
        if perplexity<min_perplexity:
            min_perplexity=perplexity
            best_model=epoch
    
    model=torch.load(f"./model{best_model}")

dataset = Corpus()
model = Model(dataset)

train(dataset, model)
# print(perplexity(dataset, model, dataset.testing_words))

if write_perplexities:
    with open("./2019114006-LMBEST-train-perplexity", "w") as f1:
        calculate_perplexity(dataset=dataset, model=model, current_data=dataset.words[:30000], savetofile=True,filename=f1)
    with open("./2019114006-LMBEST-validate-perplexity", "w") as f1:
        calculate_perplexity(dataset=dataset, model=model, current_data=dataset.validation_words[:30000], savetofile=True,filename=f1)
    with open("./2019114006-LMBEST-test-perplexity", "w") as f1:
        calculate_perplexity(dataset=dataset, model=model, current_data=dataset.testing_words[:30000], savetofile=True,filename=f1)
else:
    dat=input("ENTER SENTENCE")
    dat=dat.replace("\n", " ")
    dat=dat.replace("@", "")
    dat=dat.replace("#", "")
    dat=dat.replace("*", "")
    dat=dat.replace("+", "")
    dat=dat.replace("^", "")
    dat=dat.replace("&", "")
    dat=dat.replace("~", "")
    dat=dat.replace("  ", " ")
    dat=dat.replace("{", "")
    dat=dat.replace("}", "")
    dat=dat.replace("[", "")
    dat=dat.replace("]", "")
    dat=dat.replace("(", "")
    dat=dat.replace(")", "")
    dat=dat.replace(":", "")
    dat=dat.replace("\\", "")
    # dat=dat.replace('"', "")
    dat=dat.replace("`", "")
    sent=["<sent>"]+word_tokenize(dat)+["<sent>"]
    print(calculate_perplexity(dataset=dataset, model=model, current_data=sent, savetofile=False))
    




# perplexity=1
# print(dataset.word_to_index)
# print(predict(dataset, model, text='<sent> <sent> <sent> he saw Benny.\n yet'))