from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix
#from create_episodes_vs3 import *
#from read_files_vs3 import *
from collections import Counter
import numpy as np
from fusionART_vs3 import *

import sys
from tqdm import tqdm
import time
import pickle
import pandas as pd
import pprint
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

#from more_itertools import sliced
from sklearn.utils import shuffle
import logging
import json
import ast
from ARTxtralib import *


dataframe = pd.DataFrame()
logging.basicConfig(level = logging.INFO, filename = 'mttrace_womt_0.95_rho_22jan_vs2.log', filemode = 'w')

f3Schema = [{'name':'events','compl':True,'attrib':['y1']},{'name':'group','compl':True,'attrib':['g1','g2']}]

f2Schema = [{'name':'features','compl':True,'attrib':['bedroom','kitchen','living_room','bed','in_out']},
            {'name':'timestamp','compl':True,'attrib':['t']}]

f3 = FusionART(schema=f3Schema,beta=[1.0,1.0],alpha=[0.1,0.1],gamma=[1.0,1.0],rho=[1.0,1.0],numarray=True)
f2 = FusionART(schema=f2Schema,beta=[1.0,1.0],alpha=[0.1,0.1],gamma=[1.0,1.0],rho=[1.0,1.0],numarray=True)

#f3 is stack on top of f2
f2.stackTopFusionART(f3)
#the channel representing the sequence of events ('events') is linked with the category field (F2) 
f2.linkF2TopF1BySchema(['events'])

f3.displayNetwork()
print("===========================")

f2.displayNetwork()

## a new function of automated learning for fusion ART (fusart) object that returns a boolean True
# if there's any change to the weight values larger than mindelta or a recruitment of new uncommitted code for node J
# Otherwise it returns False.
# The function accept fusart argument as the fusion ART object, J as the selected F2 node from resonance search, and mindelta to be the minimum change of the weight to consider as converged
def xAutoLearn(fusart=None, J=None, mindelta=0.00001):
    newcode = fusart.uncommitted(J)
    bw = copy.deepcopy(fusart.codes[J]['weights']) #keep the weights values before of J before learning
    fusart.autoLearn(J)
    if not newcode:
        fusart.add_usage(J)
        return any([any([abs(bw[k][i]-fusart.codes[J]['weights'][k][i]) > mindelta for i in range(len(bw[k]))]) for k in range(len(bw))])
    else:
        return True

#learning a single event sequence at f2 and the input episode (events) channel in f3
#returning the sequence pattern formed in the input channel of f3
def learnSingleEventSeq(stemart=None, epidata=None, eplen=0):
    ecnt = 0
    arttop = stemart.TopFusionART
    arttop.clearActivityF1()
    normtime_1 = []
    for event in epidata['events']:
        normtime = 0
        if ecnt > 0:
            normtime = ecnt/eplen          
        stemart.updateF1bySchema([{'name':'features', 'val':event},
                                 {'name':'timestamp', 'val':[normtime]}])
        J = stemart.SchemaBasedSequentialResSearch(duprep=False) #do resonance search for F2 to select an event
        normtime_1.append(normtime)
        stemart.autoLearn(J) #learn/store the event at the selected node J
        ecnt += 1
    #return copy.deepcopy(arttop.activityF1)
    return copy.deepcopy(arttop.F1Fields[0]['val']),[normtime_1]

#learning a single episode at f3 based on the events sequence pattern (events) directly as the input
#in a single pass (without matchtracking)
#returning the updated em-art object
def learnSingleEpisodewEvSeq(stemart=None, epidata=None, events=None, eplen=0, foldidx=None):
    arttop = stemart.TopFusionART
    arttop.clearActivityF1()
    #arttop.updateF1bySchema([{'name':'events', 'val': events+[0]*(len(arttop.F1Fields[0]['val'])-len(events))},
    #        {'name':'group', 'val':epidata['label']}]) #assigned the episode with label (cognitive status group)
    
    #==the input to events field doesn't follow the normal complemented code for the sequence===
    arttop.updateF1bySchema([{'name':'group', 'val':epidata['label']}]) #input to label is made as normal
    valevents = events+[0]*(len(arttop.F1Fields[0]['val'])-len(events)) #special complement coding is set for events
    arttop.updateF1bySchema([{'name':'events', 'val': valevents, 'vcompl': [(0 if v <= 0 else (1-v)) for v in valevents]}],refresh=False)
    
    JJ = arttop.resSearch() 
    
    result = xAutoLearn(fusart=arttop, J=JJ) #learn/store the episode JJ 
    logging.info(f"\n fold idx: {foldidx}")
    logging.info(f"\n resultxAutolearn: {result}")
    return stemart,copy.deepcopy(arttop.activityF1[0])

#learning a single episode at f3 based on the events sequence pattern (events) directly as the input
#in multiple epochs dynamically with matchtracking
#returning True/False if there's an update in the network (weights or codes)
def learnSingleEpisodewEvSeq_topmtrack(stemart=None, epidata=None, events=None, eplen=0):
    arttop = stemart.TopFusionART
    arttop.clearActivityF1()
    #arttop.updateF1bySchema([{'name':'events', 'val': events+[0]*(len(arttop.F1Fields[0]['val'])-len(events))},
    #        {'name':'group', 'val':epidata['label']}]) #assigned the episode with label (cognitive status group)
    #==the input to events field doesn't follow the normal complemented code for the sequence===
    arttop.updateF1bySchema([{'name':'group', 'val':epidata['label']}]) #input to label is made as normal
    valevents = events+[0]*(len(arttop.F1Fields[0]['val'])-len(events)) #special complement coding is set for events
    arttop.updateF1bySchema([{'name':'events', 'val': valevents, 'vcompl': [(0 if v <= 0 else (1-v)) for v in valevents]}],refresh=False)
    
    JJ = arttop.resSearch(mtrack=[0]) #matchtracking at the first episode channel (channel index=0)
    #print('reset number:', arttop.lastResetNo) #in case for debugging
    if arttop.perfmismatch:
        print(f"PERFECT MISMATCH AT {JJ}")
    else:
        return xAutoLearn(fusart=arttop, J=JJ)
    return False


#function to learn a single episode in em-art
def learnSingleEpisode(stemart=None, epidata=None, eplen=0):
    ecnt = 0
    arttop = stemart.TopFusionART
    arttop.clearActivityF1()
    for event in epidata['events']:
        normtime = 0
        if ecnt > 0:
            normtime = ecnt/eplen
        stemart.updateF1bySchema([{'name':'features', 'val':event},
                                 {'name':'timestamp', 'val':[normtime]}])
        J = stemart.SchemaBasedSequentialResSearch() #do resonance search for F2 to select an event
        stemart.autoLearn(J) #learn/store the event at the selected node J
        ecnt += 1
    arttop.updateF1bySchema([{'name':'group', 'val':epidata['label']}]) #assigned the episode with label (cognitive status group)
    JJ = arttop.resSearch() #arttop.resSearch(resetlimit=1, outresetcnt=True)
    #print('reset number:', arttop.lastResetNo) #in case for debugging
    
    arttop.autoLearn(JJ) #learn/store the episode JJ 
    return stemart

#function to learn a single episode in em-art with matchtracking at the top fusion art
#returning True/False if there's an update in the network (weights or codes)
def learnSingleEpisode_topmtrack(stemart=None, epidata=None, eventList=None, eplen=0):
    ecnt = 0
    arttop = stemart.TopFusionART
    arttop.clearActivityF1()
    arttop.updateF1bySchema([{'name':'events', 'val':epidata['label']}]) #assigned the episode with label (cognitive status group)
    arttop.updateF1bySchema([{'name':'group', 'val':epidata['label']}]) #assigned the episode with label (cognitive status group)
    JJ = arttop.resSearch(mtrack=[0]) #matchtracking at the first episode channel (channel index=0)
    #print('reset number:', arttop.lastResetNo) #in case for debugging
    if arttop.perfmismatch:
        print(f"PERFECT MISMATCH AT {JJ}")
    else:
        return xAutoLearn(fusart=arttop, J=JJ)
    #arttop.autoLearn(JJ) #learn/store the episode JJ 
    #return stemart
    return False


#function to predict a single episode in em-art
def predictSingleEpisode(stemart=None, epidata=None, eplen=0, foldidx=None):
    ecnt = 0
    arttop = stemart.TopFusionART
    arttop.clearActivityF1()
    duplEvent = 0
    normalized_time = []
    for event in epidata['events']:
        normtime = 0
        if ecnt > 0:
            normtime = ecnt/eplen
            normalized_time.append(normtime)
            #print(f'event {ecnt} normtime {normtime}')
        stemart.updateF1bySchema([{'name':'features', 'val':event},
                                 {'name':'timestamp', 'val':[normtime]}])
        J = stemart.SchemaBasedSequentialResSearch(duprep=False) #do resonance search for F2 to select an event
        
        #if stemart.seqduplicate:
            #duplEvent += 1
        ecnt += 1
    #print(f'duplicated {duplEvent}')
    f2_activation = arttop.activityF1[0]
    JJ = arttop.resSearch() 
    print(f'code selected {JJ}, uncommitted:{arttop.uncommitted(JJ)}')
    arttop.doReadout(JJ,1)
    arttop.TopDownF1()
    pred = arttop.add_predict(JJ)
    routgroup = []
    gtruthgroup = []
    #routgroup = copy.deepcopy(arttop.F1Fields[1]['val'])
    #gtruthgroup = copy.deepcopy(epidata['label'])
    routgroup.append(arttop.F1Fields[1]['val'])
    gtruthgroup.append(epidata['label'])
    y = []
    if(arttop.F1Fields[1]['val'] == epidata['label']):
         y = arttop.add_correct(J) 
    #logging.info(f"\n fold idx: {foldidx}")
    # print(foldidx)
    #logging.info(f"\n add_predict_output: {pred}")
    # print(pred)
    #logging.info(f"\n add_correct_output: {y}")
    # print(y)
    print("pred:",routgroup)
    print("test:",gtruthgroup)
    return routgroup, gtruthgroup, (routgroup == gtruthgroup), f2_activation, [normalized_time], y, pred


all_fold_result = []
y_pred = []
y_test = []  
y_correct = []
addcorrect = []
addpredict = []
f2.setParam('rho',[0.95,0.95])
f2.setParam('gamma',[1.0,1.0])

f3.setParam('rho',[0.9,0.9]) 
f3.setParam('gamma',[1.0,1.0]) 
epilen = 96

episodeList = []
training_num_codes_f2 = [] 
norm_time_training = []

#with open('sinew_data_kfold_without_missing_feature.pickle', 'rb') as handle:
with open('Mobility_data_kfold_without_missing_features.pickle', 'rb') as handle:
      
    kfold_class_df = pickle.load(handle)
foldNum = 10
eventlist = []
label = []
print(len(kfold_class_df))
for foldidx in range(foldNum):
    eventlist = []
    count = 0
    for i in tqdm(range(len(kfold_class_df['train']['X']['fill_00'][foldidx]))):
      count += 1
      #if(count < 400):
      for v in tqdm(range(len(kfold_class_df['train']['X']['fill_00'][foldidx].iloc[i]))):
         if(count < 3001):
            events = ast.literal_eval(kfold_class_df['train']['X']['fill_00'][foldidx].iloc[i][v])
            labels = ast.literal_eval(kfold_class_df['train']['y'][foldidx].iloc[i][0])
            if(events != []):           
                 if(labels != []):
                    episode = {"events": events, "label": labels}
                    epilength = len(episode)  

                    eventlist.append(episode)
                    label.append(labels)
    print(len(label))
    print(eventlist)
    print(kfold_class_df['train']['y'][0].iloc[2][0])
    print(kfold_class_df['train']['y'][0].iloc[5])
    for i in tqdm(range(len(eventlist))):
        if(('events', 0) not in eventlist[i].items()):
            if(('label', 0) not in eventlist[i].items()):
                epilength = len(eventlist[i])
                print(f'training f2 -- fold {foldidx}, train episode {i},  episode length {epilength}')
                seqevents, normtime = learnSingleEventSeq(stemart=f2, epidata=eventlist[i], eplen=epilen)
                episodeList.append(seqevents)
                training_num_codes_f2.append(([i,len(f2.codes)]))
        else:
            episodeList.append([])
    print(len(eventlist))
    f2.displayNetwork()
    with open('eventlist2_0.95_23jan_f2_vs3.txt', 'w') as f:
            for line in eventlist:
                f.write(f"{line}\n")
    jd = json.dumps(episodeList)
            #f3 is re-stacked on f2 after training f2
    f2.stackTopFusionART(f3) 
            #re-linking f2 with the sequence of events ('events') in f3 
    f2.linkF2TopF1BySchema(['events'])
            #training for episodes in f3
            #by directly inputting the sequence patterns to f3 from episodeList
    training_num_codes_f3 = []  
    f2_act_training = []
    trainingeventlist = []
    count = 0
    for i in tqdm(range(len(kfold_class_df['train']['X']['fill_00'][foldidx]))):
      count += 1
      for v in tqdm(range(len(kfold_class_df['train']['X']['fill_00'][foldidx].iloc[i]))):

        if(count < 3001):
            events = ast.literal_eval(kfold_class_df['train']['X']['fill_00'][foldidx].iloc[i][v])
            labels = ast.literal_eval(kfold_class_df['train']['y'][foldidx].iloc[i][0])
            if(events != []):
                if(labels != []):
                        episode = {"events": events, "label": labels}
                        epilength = len(episode)  
                        print(episode)                                                                                                                                
                        trainingeventlist.append(episode)
 
                
    #f3.reset_all_cf()
    for i in tqdm(range(len(trainingeventlist))):
        if(('events', 0) not in trainingeventlist[i].items()):
            if(('label', 0) not in trainingeventlist[i].items()):
                epilength = len(trainingeventlist[i])  
                print(f'training f3 -- fold {foldidx}, train episode {i},  episode length {epilength}')
                st, f2_activation = learnSingleEpisodewEvSeq(stemart=f2, epidata=trainingeventlist[i],events=episodeList[i],eplen=epilen,foldidx=foldidx)     
                f2_act_training.append(f2_activation)
                training_num_codes_f3.append(([i,len(f3.codes)]))
        else:
            episodeList.append([])




    df5 = pd.DataFrame({'f2_activation_training':f2_act_training})
    df5.to_csv('f2_activation_training_0.95_fold_' + str(foldidx) + '.txt', sep='\t', index=False)
    #save the model and proceed with the testing
    saveFusionARTNetwork(f2, name='f2stemart_all_folds_'+ str(foldidx) + '.net') #save/dump the content of 'bottom' field fusionART

    f2.setParam('rho',[0,0]) 
    f2.setParam('gamma',[1,1])
        
    # f3.setParam('rho',[0,0])
    # f3.setParam('gamma',[1,0])
    f3.setParam('rho',[1,1])
    f3.setParam('gamma',[1,1])
    Labels = []
    Events = []
    f2_act_prediction=[]
    norm_time = []
    count=[]
    f3.reset_all_cf()
    f3.displayNetwork() # check if f3 weights coresponds to the f2 activation code
    #saveFusionARTNetwork(f3, name='f3stemart_14jan.net') #save/dump the content of 'bottom' field fusionART
    with open('eventlist2_0.95.txt', 'w') as f:
        for line in episodeList:
            f.write(f"{line}\n")
    testingeventlist = []   
    count = 0
    for i in tqdm(range(len(kfold_class_df['testing']['X']['fill_00'][foldidx]))):
      count += 1
      for v in tqdm(range(len(kfold_class_df['testing']['X']['fill_00'][foldidx].iloc[i]))):
        #if(count < 400):
            events = ast.literal_eval(kfold_class_df['testing']['X']['fill_00'][foldidx].iloc[i][v])

            label = ast.literal_eval(kfold_class_df['testing']['y'][foldidx].iloc[i][0])
            if(events != []):
                if(label != []):
                    episode = {"events": events, "label": label}
                    epilength = len(episode)  
                    print(episode)
                    Events.append(events)
                    testingeventlist.append(episode)
            

    for i in tqdm(range(len(testingeventlist))):
        if(('events', 0) not in testingeventlist[i].items()):
            if(('label', 0) not in testingeventlist[i].items()):
                epilength = len(testingeventlist[i])  
                print(epilength)
                print(f'testing  -- fold {foldidx}, test episode {i},  episode length {epilength}')
                pred, test, correct, f2_activation, normalized_time, add_correct,add_predict  = predictSingleEpisode(stemart=f2, epidata= testingeventlist[i],eplen=epilen,foldidx=foldidx)  
                # y_pred.append([''.join(str(x)) for x in pred])
                # y_test.append([''.join(str(x)) for x in test])
                y_pred.append(pred)
                y_test.append(test)
                y_correct.append(correct)
                f2_act_prediction.append(f2_activation)
                addcorrect.append(add_correct)
                addpredict.append(add_predict)
                Labels.append(label)
                norm_time.append(normalized_time)
                with open("0.95_woMT_no_of_f2_" +str(foldidx)+ ".txt", "w") as text_file3:
                    text_file3.write(str(training_num_codes_f2))
                with open("0.95_woMT_no_of_f3_" +str(foldidx)+ ".txt", "w") as text_file4:
                    text_file4.write(str(training_num_codes_f3))
            else:
                episodeList.append([])
        else:
            episodeList.append([])
    #saveFusionARTNetwork(f3, name='fusionART_without_match_tracking'+str(foldidx)+'.net')
    saveFusionARTNetwork(f3, name='f3stemart_all_folds_'+ str(foldidx) + '.net')
    print(y_pred)
    print(y_test)
    df6 = pd.DataFrame({'f2_activation_at_prediction':f2_act_prediction})
    df6.to_csv('f2_activation_at_prediction_0.95_fold_' + str(foldidx) + '.txt', sep='\t', index=False)
    with open('eventlist3_0.95.txt', 'w') as f:
                for line in testingeventlist:
                    f.write(f"{line}\n") 
    #print(y_test)
    print(len(episodeList))
    print(len(testingeventlist))
    print(trainingeventlist)
    import itertools
    # newtest = list(np.concatenate(y_test))
    # newpred = list(np.concatenate(y_pred)) 
    y_tests = list(itertools.chain.from_iterable(y_test))

    y_tests = [item[0] for item in y_tests]
    y_preds = list(itertools.chain.from_iterable(y_pred))

    y_preds = [item[0] for item in y_preds]
    print(y_pred)
    from sklearn.metrics import multilabel_confusion_matrix
    from sklearn.metrics import classification_report

    mcm = multilabel_confusion_matrix(y_tests, y_preds, labels=[0, 1])
    tps = mcm[:, 1, 1]
    tns = mcm[:, 0, 0]
    precision = tps / (tps + mcm[:, 0, 1])
    specificity = tns / (tns + mcm[:, 0, 1])         
    print(f"specificity class MCI: {specificity[0]}. specificity class NC: {specificity[1]}")
    print(classification_report(y_tests, y_preds, labels=[0, 1]))

    report = classification_report(y_tests, y_preds, labels=[0,1], output_dict=True)
  
    CR = pd.DataFrame(report).transpose()
    CR['specificity-mci'] = specificity[0]
    CR['specificity-nc'] = specificity[1]
    
    dataframe = pd.concat([dataframe, CR]) 
    print(dataframe)
   
    CR.to_csv('update_predictions_0.95_STEM_fold_' + str(foldidx) + '.csv', index=False)
dataframe.to_csv('update_predictions_0.95_all_folds_STEM.csv')
