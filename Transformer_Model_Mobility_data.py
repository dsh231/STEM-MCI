import pandas as pd
import numpy as np
import re
import torch
from transformers import BertTokenizer
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import defaultdict
from sklearn import metrics
import pickle
from tqdm import tqdm
import ast
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np

with open('Mobility_data_kfold_without_missing_features.pickle', 'rb') as handle:
    kfold_class_df = pickle.load(handle)

kfold_num = 10


all_fold_result = []
kfold_num = 1
y_pred = []
y_test = []
for foldidx in range(kfold_num):
    trainingevent = []
    traininglabel = []
    testingevent = []
    testinglabel = []
    events1 = []
    training_data = []
    testing_data = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    y_pred=[]
    labels1 = []
    labels2 = []
    traintestlabel = []
    for i in tqdm(range(len(kfold_class_df['train']['X']['fill_00'][foldidx]))):
       for v in tqdm(range(len(kfold_class_df['train']['X']['fill_00'][foldidx].iloc[i]))):
           events = ast.literal_eval(kfold_class_df['train']['X']['fill_00'][foldidx].iloc[i][v])
           labels = ast.literal_eval(kfold_class_df['train']['y'][foldidx].iloc[i][v])
           train = []
           if(labels[0] == 1):
            events = list(chain.from_iterable(events))
            events1 = list(map(int, events))
            events1.append(0)
            events1.append(0)

           else:
            events = list(chain.from_iterable(events))
            events1 = list(map(int, events))
            events1.append(0)
            events1.append(1)
           trainingevent.append(events1)

    for i in tqdm(range(len(kfold_class_df['testing']['X']['fill_00'][foldidx]))):
       for v in tqdm(range(len(kfold_class_df['testing']['X']['fill_00'][foldidx].iloc[i]))):
           events = ast.literal_eval(kfold_class_df['testing']['X']['fill_00'][foldidx].iloc[i][v])
           labels = ast.literal_eval(kfold_class_df['testing']['y'][foldidx].iloc[i][v])
           test = []
           if(labels[0] == 1):
              events = list(chain.from_iterable(events))
              events1 = list(map(int, events))
              events1.append(0)
              events1.append(0)
              

           else:
              events = list(chain.from_iterable(events))
              events1 = list(map(int, events))
              events1.append(0)
              events1.append(1)
             
              
           testingevent.append(events1)
                    
    print(trainingevent)
    train_data = trainingevent
    train_df_2 = pd.DataFrame(train_data)
    
    eval_data = testingevent
    eval_df_2 = pd.DataFrame(eval_data)
    from numpy.random import default_rng

    arr_indices_top_drop = default_rng().choice(train_df_2.index, size=0, replace=False)
    train_df_2 = train_df_2.drop(index=arr_indices_top_drop)


    train_df_2.to_csv('train_df.txt',header=False,index=False)

    arr_indices_top_drop = default_rng().choice(eval_df_2.index, size=0, replace=False)
    eval_df_2 = eval_df_2.drop(index=arr_indices_top_drop)
    eval_df_2.to_csv('eval_df.txt',header=False,index=False)

    with open('train_df.txt', "r+", encoding="utf-8") as csv_file:
        content = csv_file.read()

    with open('train_df.txt', "w+", encoding="utf-8") as csv_file:
        csv_file.write(content.replace('"', ''))

    df = pd.read_csv('train_df.txt')
    #num_bins = len(df) // 5
    remainder = len(df) % 10
    df1 = df[remainder:]
    df1.to_csv('train_df.txt',header=False,index=False)
    print(len(df))

    with open('eval_df.txt', "r+", encoding="utf-8") as csv_file:
        content = csv_file.read()

    with open('eval_df.txt', "w+", encoding="utf-8") as csv_file:
        csv_file.write(content.replace('"', ''))

    df = pd.read_csv('eval_df.txt')
    #num_bins = len(df) // 5
    remainder = len(df) % 10
    df2 = df[remainder:]
    df2.to_csv('eval_df.txt',header=False,index=False)

    
  

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import numpy
    torch.manual_seed(1234)
    import torch.utils.data as data_utils
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_size = 20000

    block_size = 500
    embeds_size = 482
    num_classes = 2
    drop_prob = 0.13
    batch_size = 30
    epochs = 20
    num_heads = 2
    head_size = embeds_size // num_heads
    model_path = 'model_classification.pth'
    model_loader = False
    class DataSet(Dataset):
        def __init__(self, src_file):
            all_xy = np.loadtxt(src_file, usecols=range(0,482),
            delimiter=",", comments="#", dtype=int)
            #all_xy = src_file
            tmp_x = all_xy[:,0:481]   # cols [0,6) = [0,5]
            tmp_y = all_xy[:,481]     # 1-D

            self.x_data = torch.tensor(tmp_x, dtype=int).to(device)
            self.y_data = torch.tensor(tmp_y,dtype=int).to(device)  # 1-D
            print(len(self.y_data))

        def __len__(self):
            return len(self.x_data)

        def __getitem__(self, idx):
            preds = self.x_data[idx]
            trgts = self.y_data[idx] 
            return preds, trgts  # as a Tuple

    train_file = "train_df.txt"
    train_ds = DataSet(train_file)

    test_file = "eval_df.txt"
    test_ds = DataSet(test_file)
    train_data = torch.utils.data.DataLoader(train_ds, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_ds, shuffle=False)
    y_pred = []
    y_test = []


    class block(nn.Module):
        def __init__(self):
            super(block, self).__init__()
            self.attention = nn.MultiheadAttention(embeds_size, num_heads, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(embeds_size, 2 * embeds_size),
                nn.LeakyReLU(),
                nn.Linear(2 * embeds_size, embeds_size),
            )
            self.drop1 = nn.Dropout(drop_prob)
            self.drop2 = nn.Dropout(drop_prob)
            self.ln1 = nn.LayerNorm(embeds_size)
            self.ln2 = nn.LayerNorm(embeds_size)

        def forward(self, hidden_state):
            attn, _ = self.attention(hidden_state, hidden_state, hidden_state, need_weights=False)
            attn = self.drop1(attn)
            out = self.ln1(hidden_state + attn)
            observed = self.ffn(out)
            observed = self.drop2(observed)
            return self.ln2(out + observed)


    class transformer(nn.Module):
        def __init__(self):
            super(transformer, self).__init__()
            self.tok_emb = nn.Embedding(vocab_size, embeds_size)
            self.pos_emb = nn.Embedding(block_size, embeds_size)
            self.block = block()
            self.ln1 = nn.LayerNorm(embeds_size)
            self.ln2 = nn.LayerNorm(embeds_size)
            self.classifier_head = nn.Sequential(
                nn.Linear(481, 481),
                nn.LeakyReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(481, 481),
                nn.LeakyReLU(),
                nn.Linear(481, 1),
            )

            print("number of parameters: %.2fM" % (self.num_params()/1e6,))

        def num_params(self):
            n_params = sum(p.numel() for p in self.parameters())
            return n_params

        def forward(self, seq):
            B,T = seq.shape
            embedded = self.tok_emb(seq)
            embedded = embedded + self.pos_emb(torch.arange(T, device=device))
            embedded = self.pos_emb(torch.arange(T, device=device))
            output = self.block(embedded)
            output = output.mean(dim=1)
            output = self.classifier_head(output)
            return output


    model = transformer()
    if model_loader:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model_loss = nn.BCEWithLogitsLoss()
    model_optimizer = torch.optim.RMSprop(model.parameters(), lr=4e-2)

    for epoch in range(epochs):
        progress_bar = tqdm(train_data,
                            desc='Epoch {:1d}'.format(epoch),
                            leave=False,
                            disable=False)
        losses = 0
        for (inputs,targets) in progress_bar: 
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            print(targets)
            loss = model_loss(output, targets.float())
            print(output)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            losses += loss.item()
        print(f'[{epoch}][Train]', losses)
        model.train()
        test_loss = 0
        passed = 0
        progress_bar = tqdm(test_data,
                            desc='Epoch {:1d}'.format(epoch),
                            leave=False,
                            disable=False)
        for (inputs, targets) in progress_bar:
           # with torch.no_grad():
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                y_test.append(targets)
                #print(inputs)
                print(targets)
                print(output)
                #loss = model_loss(output, targets.float())
                #print(output)
                output = output.detach().numpy()
                #print(output)
                y_pred.append(output)
                
                # targets = list(map(lambda el:[el], targets))
                if output.argmax() == targets.argmax():
                    passed += 1
                
        #model.train()
        #print(f'[{epoch}][Test]', ', accuracy', passed / len(test_ds))
    df5 = pd.DataFrame({'col':y_test})
    df6 = pd.DataFrame({'col':y_pred})
    df5.to_csv('y_test_transformer.csv')
    df6.to_csv('y_pred_transformer.csv')
    
    torch.save(model.state_dict(), model_path)
    import pandas as pd  
    import pandas as pd  
    from sklearn.metrics import precision_recall_fscore_support as score
    from collections import Counter
    from sklearn.metrics import classification_report
    from sklearn.metrics import multilabel_confusion_matrix
    dataframe = pd.DataFrame()
        
    # making dataframe  
    df = pd.read_csv("y_pred_transformer.csv")  
    
    # output the dataframe 
    print(df)
    df['cols'] = df["col"].str[1:-1]
    df['cols'] = df['cols'].astype(float)

    print(df)
    df.drop(["col"], axis=1, inplace=True)
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    print(df)
    normalized_df=(df-df.min())/(df.max()-df.min())
    print(normalized_df)
    #normalized_df.to_csv('/home/shantho/Anothe_project/CNN-Bi-LSTM_Model/Multilabel-Text-Classification-using-novel-CNN-Bi-LSTM-framework-main/transformer_pred.csv')
    normalized_df.loc[normalized_df['cols'] > 0.5, 'cols'] = 1.0
    normalized_df.loc[normalized_df['cols'] <= 0.5, 'cols'] = 0.0
    normalized_df['cols'] = normalized_df['cols'].astype('float')
    y_test = pd.read_csv("y_test_transformer.csv")  
    y_test['cols'] = y_test['col'].str[8:9]
    y_test.drop(["col"], axis=1, inplace=True)
    y_test.drop(["Unnamed: 0"], axis=1, inplace=True)
    y_test['cols'] = y_test['cols'].astype('float')
    y_test = y_test['cols'].tolist()
    y_pred = normalized_df['cols'].tolist()
    print(y_test)
    print(y_pred)
    target_names = ['0.0', '1.0']
    report = classification_report(y_pred, y_test, target_names=target_names, zero_division=0)
    print(report)
    report = classification_report(y_test, y_pred, labels=[0.0,1.0], output_dict=True)
    cm = multilabel_confusion_matrix(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
    CR = pd.DataFrame(report).transpose()
    CR['specificity-mci'] = specificity1[0]
    CR['specificity-nc'] = specificity1[1]
    CR['AUC'] = auc
    dataframe = pd.concat([dataframe, CR])
    print(dataframe)       
    

dataframe.to_csv('classification_report_Transformer_Mobility_Dataset.csv') 