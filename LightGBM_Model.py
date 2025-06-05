import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
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
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import ast
import sklearn.metrics as skm
import pickle
dataframe = pd.DataFrame()



with open('Mobility_data_kfold_without_missing_features.pickle', 'rb') as handle:
      
     kfold_class_df = pickle.load(handle)
  
  
kfold_num = 1
all_fold_result = []

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
    for i in tqdm(range(len(kfold_class_df['train']['X']['fill_00'][foldidx]))):
           events = kfold_class_df['train']['X']['fill_00'][foldidx].values[i].tolist()
           labels = kfold_class_df['train']['y'][foldidx].values[i].tolist()
           #single_train = []
           #single_train.append(events)
           if(labels[0] == 0):
            events.append(0.0)
           elif(labels[0] > 0):
            events.append(1.0)
           x_train.append(events)
               
    for i in tqdm(range(len(kfold_class_df['testing']['X']['fill_00'][foldidx]))):
           events = kfold_class_df['testing']['X']['fill_00'][foldidx].values[i].tolist()
           labels = kfold_class_df['testing']['y'][foldidx].values[i].tolist()
           #single_test = []
           #single_test.append(events)
           if(labels[0] == 0):
                events.append(0.0)
           elif(labels[0] > 0):
                events.append(1.0)
           x_test.append(events)
 
    from lightgbm import LGBMClassifier
    import lightgbm as lgb

    params = {
    'objective': 'multiclass',
    'num_class': 2,
    'metric': 'multi_logloss',
    'verbose': 0
    }
    single_train_y = [sublist[-1] for sublist in x_train]
    single_test_y = [sublist[-1] for sublist in x_test]


  
    single_train_x = [x[:-1] for x in x_train]  

    
    single_test_x = [x[:-1] for x in x_test]  
    print(single_train_x)
    num_round = 100
    clf = lgb.LGBMClassifier()
    clf.fit(single_train_x, single_train_y)
    y_pred=clf.predict(single_test_x)

    print(y_pred)
    print(len(y_pred))
    print(len(single_test_y))
    
    tn, fp, fn, tp = confusion_matrix(single_test_y, y_pred).ravel()
    precision_score = tp / (tp + fp)
    recall_score = tp / (tp + fn)

    cm = multilabel_confusion_matrix(single_test_y, y_pred)
    specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
    RF_fold_result  = pd.DataFrame()
    RF_fold_result['specificity-mci'] = specificity1[0]
    RF_fold_result['specificity-nc'] = specificity1[1]
   
    all_fold_result.append(RF_fold_result)

    report = classification_report(single_test_y, y_pred,output_dict=True)
    auc = metrics.roc_auc_score(single_test_y, y_pred)
    CR = pd.DataFrame(report).transpose()
    CR['specificity-mci'] = specificity1[0]
    CR['specificity-nc'] = specificity1[1]
    CR['AUC'] = auc
    CR['train_num'] = len(kfold_class_df['train']['X']['fill_00'][foldidx])
    CR['test_num'] = len(single_test_y)
    dataframe = pd.concat([dataframe, CR])
    print(dataframe)
    print(clf.get_params())
dataframe.to_csv('classification_report_lightgbm.csv')
