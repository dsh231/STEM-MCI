import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from collections import Counter
from sklearn.metrics import classification_report
from itertools import chain
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import sklearn.metrics as skm
from numpy import array
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
import math
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.engine import data_adapter
dataframe = pd.DataFrame()
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# You would just need to add the code below to yours
import tensorflow as tf
print(tf.__version__)
from tensorflow.python.framework.versions import VERSION

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import keras
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
import ast
import pickle

#------------------------------------------------for SINEW Dataset--------------------------------------------------------------
with open('Mobility_data_kfold_without_missing_features.pickle', 'rb') as handle:
      
    kfold_class_df = pickle.load(handle)

kfold_num = 1
all_fold_result = []
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
    labels1 = []
    labels2 = []
    print(kfold_class_df['train']['X']['fill_00'][foldidx])
    print(type(kfold_class_df['train']['X']['fill_00'][0]))
    #df = pd.read_pickle("/home/shantho/fusionart/update_fold_v6.2.1_" + str(foldidx) + ".pkl")
    #kfold_class_df = df
    #for i in tqdm(range(len(kfold_class_df['train']['X']['fill_11']))):
    for i in tqdm(range(len(kfold_class_df['train']['X']['fill_00'][foldidx]))):
       for v in tqdm(range(len(kfold_class_df['train']['X']['fill_00'][foldidx].iloc[i]))):
           #list1 = kfold_class_df['train']['X']['events'][foldidx].iloc[i][v].values()
           #res = pd.array(kfold_class_df['train']['X']['events'][foldidx].iloc[i][v], dtype=np.float64)
           #res = kfold_class_df['train']['X']['events'][foldidx].iloc[i][v].apply(lambda x: np.fromstring(x))
           events = ast.literal_eval(kfold_class_df['train']['X']['fill_00'][foldidx].iloc[i][v])
           labels = ast.literal_eval(kfold_class_df['train']['y'][foldidx].iloc[i][v])
           if(events != []):
                if(labels != []):
                    # single_train = list(chain.from_iterable(events))
                    # x_train.append(single_train)
                    x_train.append(events)
                    if(labels[0] > 0.0):
                        y_train.append(0.0)
                    elif(labels[1] > 0.0):
                        y_train.append(1.0)
    for i in tqdm(range(len(kfold_class_df['testing']['X']['fill_00'][foldidx]))):
       for v in tqdm(range(len(kfold_class_df['testing']['X']['fill_00'][foldidx].iloc[i]))):
           #list1 = kfold_class_df['train']['X']['events'][foldidx].iloc[i][v].values()
           #res = pd.array(kfold_class_df['train']['X']['events'][foldidx].iloc[i][v], dtype=np.float64)
           #res = kfold_class_df['train']['X']['events'][foldidx].iloc[i][v].apply(lambda x: np.fromstring(x))
           events = ast.literal_eval(kfold_class_df['testing']['X']['fill_00'][foldidx].iloc[i][v])
           labels = ast.literal_eval(kfold_class_df['testing']['y'][foldidx].iloc[i][v])
           if(events != []):
                if(labels != []):
                    # single_train = list(chain.from_iterable(events))
                    # x_test.append(single_train)
                    x_test.append(events)
                    if(labels[0] > 0.0):
                        y_test.append(0.0)
                    elif(labels[1] > 0.0):
                        y_test.append(1.0)


    

    X_train = np.array(x_train)
    Y_train = np.array(y_train) 
    array_3d = X_train.reshape((len(X_train), 480, 1)) 
    array_3d_y = np.repeat(Y_train[:, np.newaxis, np.newaxis], 480, axis=1)
    X_test = np.array(x_test)    
    array_3d_x_test = X_test.reshape((len(X_test), 480, 1)) 
    y_test = np.array(y_test)          
    array_3d_y_test = np.repeat(y_test[:, np.newaxis, np.newaxis], 480, axis=1)             
    print(len(X_train))
    print(len(X_test))
 


    lst = Sequential() # initializing model

    # input layer and LSTM layer with 50 neurons
    # outpute layer with sigmoid activation
    lst.add(LSTM(units=96, dropout=0.7, recurrent_dropout=0.7))
    lst.add(Dense(1,activation='sigmoid'))
    lst.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    lst.fit(X_train, Y_train, epochs = 11)


    print(X_test[:2:].shape)
    print(len(X_test[0]))
    print(len(X_train[0]))
    y_pred = lst.predict(X_test)
    y_pred=np.transpose(y_pred)[0]  # transformation to get (n,)
    print(y_pred.shape)  # now the shape is (n,)
    # Applying transformation to get binary values predictions with 0.5 as thresold

    y_pred = list(map(lambda x: 0.0 if x<0.5 else 1.0, y_pred))
    mysetpred = set(y_pred)
    count_1 = np.unique(y_pred, return_counts=True)
    print(y_pred)
    print(mysetpred)
    print(count_1)
    report = classification_report(y_test, y_pred, labels=[0.0,1.0], output_dict=True)
    cm = multilabel_confusion_matrix(y_test, y_pred)
    specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
    CR = pd.DataFrame(report).transpose()
    CR['specificity-mci'] = specificity1[0]
    CR['specificity-nc'] = specificity1[1]
    dataframe = pd.concat([dataframe, CR])
    mcm = multilabel_confusion_matrix(y_test, y_pred)

    tps = mcm[:, 1, 1]
    tns = mcm[:, 0, 0]

    recall      = tps / (tps + mcm[:, 1, 0])         # Sensitivity
    specificity = tns / (tns + mcm[:, 0, 1])         # Specificity
    precision   = tps / (tps + mcm[:, 0, 1])  
    print(mcm)
    print(specificity)
    print(classification_report(y_test, y_pred, labels=[0.0,1.0]))
    print(precision_recall_fscore_support(y_test, y_pred, average=None,labels=[0.0,1.0]))
dataframe.to_csv('classification_report_LSTM.csv')


