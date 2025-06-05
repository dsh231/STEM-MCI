import pandas as pd
import numpy as np
from tqdm import tqdm
import json  
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from collections import Counter
from sklearn.metrics import classification_report
from itertools import chain
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from array import array
from sklearn import svm
import ast 
import sklearn.metrics as skm
import itertools
import pickle
dataframe = pd.DataFrame()

with open('Mobility_data_kfold_without_missing_features.pickle', 'rb') as handle:
      
    kfold_class_df = pickle.load(handle)

kfold_num = 10

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
    for i in tqdm(range(len(kfold_class_df['train']['X']['fill_00'][foldidx]))):
       for v in tqdm(range(len(kfold_class_df['train']['X']['fill_00'][foldidx].iloc[i]))):
           events = ast.literal_eval(kfold_class_df['train']['X']['fill_00'][foldidx].iloc[i][v])
           labels = ast.literal_eval(kfold_class_df['train']['y'][foldidx].iloc[i][v])
           if(events != []):
                if(labels != []):
                    single_train = list(chain.from_iterable(events))
                    x_train.append(single_train)
                    if(labels[0] > 0.0):
                        y_train.append(0.0)
                    elif(labels[1] > 0.0):
                        y_train.append(1.0)
    for i in tqdm(range(len(kfold_class_df['testing']['X']['fill_00'][foldidx]))):
       for v in tqdm(range(len(kfold_class_df['testing']['X']['fill_00'][foldidx].iloc[i]))):
           events = ast.literal_eval(kfold_class_df['testing']['X']['fill_00'][foldidx].iloc[i][v])
           labels = ast.literal_eval(kfold_class_df['testing']['y'][foldidx].iloc[i][v])
           if(events != []):
                if(labels != []):
                    single_train = list(chain.from_iterable(events))
                    x_test.append(single_train)
                    if(labels[0] > 0.0):
                        y_test.append(0.0)
                    elif(labels[1] > 0.0):
                        y_test.append(1.0)
    
    myset = set(y_test)
    #print(myset)
    lsvm = svm.SVC()
    lsvm.fit(x_train, y_train)
    score = lsvm.score(x_test, y_test)
    y_pred = lsvm.predict(x_test) 
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision_score = tp / (tp + fp)
    recall_score = tp / (tp + fn)

    #svm_fold_result = pd.DataFrame(precision_recall_fscore_support(y_test, y_pred, average='None', labels=[0,1]),index=['precision','recall','F1','support'])
    cm = multilabel_confusion_matrix(y_test, y_pred)
    specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
    svm_fold_result  = pd.DataFrame()
    svm_fold_result['specificity-mci'] = specificity1[0]
    svm_fold_result['specificity-nc'] = specificity1[1]
   
    all_fold_result.append(svm_fold_result)
    report = classification_report(y_test, y_pred,output_dict=True)
    CR = pd.DataFrame(report).transpose()
    CR['specificity-mci'] = specificity1[0]
    CR['specificity-nc'] = specificity1[1]
    CR['train_num'] = len(kfold_class_df['train']['X']['fill_00'][foldidx])
    CR['test_num'] = len(y_test)
    dataframe = pd.concat([dataframe, CR])
    mcm = multilabel_confusion_matrix(y_test, y_pred)

    tps = mcm[:, 1, 1]
    tns = mcm[:, 0, 0]

    recall      = tps / (tps + mcm[:, 1, 0])         # Sensitivity
    specificity = tns / (tns + mcm[:, 0, 1])         # Specificity
    precision   = tps / (tps + mcm[:, 0, 1])  
    print(mcm)
    print(specificity)   
dataframe.to_csv('classification_report_svm.csv')
