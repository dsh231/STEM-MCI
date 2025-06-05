# New temporary update Jan 2025
#-----------------------------------------------------------------------------------------
#last update 24 Apr 2022 -- added choiceFieldFuncART2Quick as a faster ART2A-C choice function with corresponding update on ART2ACModelOverride
#22 Oct 2021
#----------------------------
#Extra functions library for fusion ART beyond FuzzyART model

import copy
import numpy as np
import math


ROUNDDIGIT = 10 #number of digit precision for round


#--Functions for ART2 (ART2A-C)-----------------------------------------------------------

#choiceFieldFuncART2: calculate the choice function for a node j in field k
#given input xck, a weight vector wjck, alphack, and gammack for contribution parameter and choice parameters 
#return the activation value of node j for particular field k
#The template for ART2 choice function for a particular channel/field can be defined as follows:
def choiceFieldFuncART2(xck,wjck,alphack,gammack):
    tp = np.dot(np.array(xck),np.array(wjck))
    btm = np.linalg.norm(np.array(xck))*np.linalg.norm(np.array(wjck))
    return gammack * (round(float(tp)/float(btm),ROUNDDIGIT))

#choiceFieldFuncART2Quick: calculate the choice function for a node j in field k
#given input xck, a weight vector wjck, alphack, and gammack for contribution parameter and choice parameters 
#return the activation value of node j for particular field k
#The template for ART2 choice function for a particular channel/field can be defined as follows:
def choiceFieldFuncART2Quick(xck,wck,alphack,gammack):
    #print('xck ', xck, ' wck ', list(wck))
    return gammack * np.dot(xck,np.array(list(wck)).T)*((np.linalg.norm(np.array(xck))*np.linalg.norm(np.array(list(wck)),axis=1))**-1.0)
    #return gammack * ((np.linalg.norm(np.array(xck))*np.linalg.norm(np.array(wck),axis=1))**-1.0)

#matchFuncART2: ART2 match function of weight vector wjck with vector xck
#return the match value. 
##The template function for ART2 template matching for a particular channel/field can defined as follows: 
def matchFuncART2(xck,wjck):
    m = 0.0
    denominator = 0.0
    tp = np.dot(np.array(xck),np.array(wjck))
    btm = np.linalg.norm(np.array(xck))*np.linalg.norm(np.array(wjck))
    if btm <= 0:
        return 1.0
    return round(float(tp)/float(btm),10)

#The template include the checking of the match with vigilance parameter
def resonanceFieldART2(xck, wjck, rhok):
    return matchFuncART2(xck, wjck) < rhok

#updWeightsART2: ART template learning function of weight vector wjck with vector xck
#return the updated weight. 
##The template function for ART2 template learning for a particular channel/field can defined as follows: 
def updWeightsART2(rate, weightk, inputk):
    w = np.array(weightk)
    i = np.array(inputk)
    uw = (((1-rate)*w) + (rate*i)).tolist()
    return uw


def ART2ACModelOverride(fa_model, k=-1):		#Override all the resonance search functions with ART2 (ART2A-C) for F1 field(s). 
	#k is the index of the channel to override 
	#set the choice function to activate category field
    #fa_model.setChoiceActFunction(cfunction=choiceFieldFuncART2, k=k)
    fa_model.setChoiceActFunction(cfunction=choiceFieldFuncART2Quick, k=k)

    #set the weight update function
    fa_model.setUpdWeightFunction(ufunction=updWeightsART2, k=k)

    #set the resonance search function
    fa_model.setResonanceFieldFunction(rfunction=resonanceFieldART2, k=k)

    #set the match function
    fa_model.setMatchValFieldFunction(mfunction=matchFuncART2, k=k)

#--------------------------------------------------------------------------------


#--------------------------------------------------------------------
def CodesUsage(fa_model, k=None, outVal=None):
    Urc = [0] * len(fa_model.codes)
    uRt = fa_model.usageRate(k)
    for j in range(len(fa_model.codes)):
        if not fa_model.uncommitted(j):
            if fa_model.doRetrieve(j,k) == outVal:
                Urc[j] = uRt[j]
    return Urc

def CodesAccuracy(fa_model, k=None, outVal=None):
    Acrc = [0] * len(fa_model.codes)
    AcRt = fa_model.accuracyRate(k)
    for j in range(len(fa_model.codes)):
        if not fa_model.uncommitted(j):
            if fa_model.doRetrieve(j,k) == outVal:
                Acrc[j] = AcRt[j]
    return Acrc

def CodesConfidence(fa_model, k=None, outVal=None, theta=0.5):
    Crc = [0] * len(fa_model.codes)
    cRt = fa_model.confidenceFactor(k, theta=theta)
    for j in range(len(fa_model.codes)):
        if not fa_model.uncommitted(j):
            if fa_model.doRetrieve(j,k) == outVal:
                Crc[j] = cRt[j]
    return Crc

def CodesUsagebySchema(fa_model, field=None, outVal=None):
    Urc = [0] * len(fa_model.codes)
    k = fa_model.getFieldIdxbySchema(field)
    uRt = fa_model.usageRate(k)
    for j in range(len(fa_model.codes)):
        if not fa_model.uncommitted(j):
            rk = fa_model.doRetrieve(j,k)
            if fa_model.F1Fields[k]['compl']:
                if rk[:int(len(rk)/2)] == outVal:
                    Urc[j] = uRt[j]
            elif rk == outVal:
                Urc[j] = uRt[j]
    return Urc

def CodesAccuracybySchema(fa_model, field=None, outVal=None):
    Acrc = [0] * len(fa_model.codes)
    k = fa_model.getFieldIdxbySchema(field)
    AcRt = fa_model.accuracyRate(k)
    for j in range(len(fa_model.codes)):
        if not fa_model.uncommitted(j):
            rk = fa_model.doRetrieve(j,k)
            if fa_model.F1Fields[k]['compl']:
                if rk[:int(len(rk)/2)] == outVal:
                    Acrc[j] = AcRt[j]
            elif rk == outVal:
                Acrc[j] = AcRt[j]
    return Acrc

def CodesConfidencebySchema(fa_model, field=None, outVal=None, theta=0.5):
    cRc = [0] * len(fa_model.codes)
    k = fa_model.getFieldIdxbySchema(field)
    cRt = fa_model.confidenceFactor(k, theta=theta)
    for j in range(len(fa_model.codes)):
        if not fa_model.uncommitted(j):
            rk = fa_model.doRetrieve(j,k)
            if fa_model.F1Fields[k]['compl']:
                if rk[:int(len(rk)/2)] == outVal:
                    cRc[j] = cRt[j]
            elif rk == outVal:
                cRc[j] = cRt[j]
    return cRc

def ConfidenceIdx(fa_model, k=None, outVal=None, theta=0.5):
    cconfids = CodesConfidence(fa_model, k=k, outVal=outVal, theta=theta)
    cconfidsort = list(np.argsort(cconfids))
    cconfidsort.reverse()
    return [{'code':cconfidsort[i],'confidence':cconfids[cconfidsort[i]]} for i in range(len(cconfidsort))]

def ConfidenceIdxbySchema(fa_model, field=None, outVal=None, theta=0.5):
    cconfids = CodesConfidencebySchema(fa_model, field=field, outVal=outVal, theta=theta)
    cconfidsort = list(np.argsort(cconfids))
    cconfidsort.reverse()
    return [{'code':cconfidsort[i],'confidence':cconfids[cconfidsort[i]]} for i in range(len(cconfidsort))]

def AccuracyIdxbySchema(fa_model, field=None, outVal=None):
    accfids = CodesAccuracybySchema(fa_model, field=field, outVal=outVal)
    accfidsort = list(np.argsort(accfids))
    accfidsort.reverse()
    return [{'code':accfidsort[i],'accuracy':accfids[accfidsort[i]]} for i in range(len(accfidsort))]

def UsageIdxbySchema(fa_model, field=None, outVal=None):
    usageids = CodesUsagebySchema(fa_model, field=field, outVal=outVal)
    usageidsort = list(np.argsort(usageids))
    usageidsort.reverse()
    return [{'code':usageidsort[i],'usage':usageids[usageidsort[i]]} for i in range(len(usageidsort))]