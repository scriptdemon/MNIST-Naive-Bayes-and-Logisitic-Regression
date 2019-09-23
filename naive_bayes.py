# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import statistics as stat
import math as m
import preprocessor as pre

def getMeanTestFeatures(txSet):
    meanSet = []
    for i in txSet:
        meanSet.append(sum(i)/float(len(i)))
    return meanSet

def getMean(tup):
    return sum(tup)/len(tup)

def getSdTestFeatures(txSet):
    sdSet = []
    for i in txSet:
        sdSet.append(stat.pstdev(i))
    return sdSet 

def getTranspose(txSet):
    return [list(x) for x in zip(*txSet)]

def getPriorProbability(tySet,label):
    return (list(tySet).count(label)/len(tySet))

def getPosteriorProbability(tsTuple,mean,sd):
    post_prob = 1
    count = 0
    for i in tsTuple:
        post_prob*=getGaussianProbability(i,mean[count],sd[count])
        count+=1
    return post_prob

def getGaussianProbability(x,mean,sd):
    a = 1/(m.sqrt(2*m.pi)*sd)
    b = m.exp(-m.pow(x-mean,2)/(2*m.pow(sd,2)))
    return a*b

def getGaussianLogProbability(x,mean,sd):
    expValue = getGaussianProbability(x,mean,sd)
    #print (m.log(expValue))
    return m.log(expValue)

def getAccuracy(tsY,predY):
    hits_7 = 0
    hits_8 = 0
    total_7 = 0
    total_8 = 0
    for i in range(len(tsY)):
        if(tsY[i] == 0):
            total_7+=1
        else:
            total_8+=1
        if (tsY[i] == predY[i]):
            if (predY[i] == 0):
                hits_7+=1
            else:
                hits_8+=1
    return hits_7/total_7,hits_8/total_8,(hits_7 + hits_8)/(total_7 + total_8)

def testNaiveBayes(tsX,prior_7,prior_8,mean_7,sd_7,mean_8,sd_8):
    #Calculating P(C)*[product(P(xi|C))] for C=7,8 for test set tsX
    pred_y = []
    for i in tsX:
        p7 = prior_7*getPosteriorProbability(i,mean_7,sd_7)
        p8 = prior_8*getPosteriorProbability(i,mean_8,sd_8)
        if p7 > p8:
            pred_y.append(0)
        else:
            pred_y.append(1)
    return pred_y    
        
def naiveBayes(trX,trY,tsX,tsY):
    # Splitting training Feature set(trxFeatures) for digits 7 and 8
    # trX_7 -> training data for which label is 0
    # trX_8 -> training data for which label is 1 
    trX_7 = pre.splitTestSetClasswise(trX,trY,0)
    trX_8 = pre.splitTestSetClasswise(trX,trY,1)
    
    # Calculating mean and variance for each feature
    # mean_7 -> mean values features for training data in trX_7
    # sd_7 -> standard deviation values of features for training data in trX_7
    # mean_8 -> mean values of features for training data in trX_8
    # sd_8 -> standard deviation values of features for training data in trX_8
    trX_7_transpose = getTranspose(trX_7)
    trX_8_transpose = getTranspose(trX_8)
    
    mean_7 = getMeanTestFeatures(trX_7_transpose)
    mean_8 = getMeanTestFeatures(trX_8_transpose)
    sd_7 = getSdTestFeatures(trX_7_transpose)
    sd_8 = getSdTestFeatures(trX_8_transpose)
    
    #Calculating prior probabilities for digits 7 and 8
    # prior_7 -> prior probability that label is 0 (i.e. digit observed is 7)
    # prior_8 -> prior probability that label is 1 (i.e. digit observed is 8)
    prior_7 = getPriorProbability(trY,0)
    prior_8 = getPriorProbability(trY,1)
    
    pred_y = testNaiveBayes(tsX,prior_7,prior_8,mean_7,sd_7,mean_8,sd_8)
    accuracy_7,accuracy_8,accuracy_total = getAccuracy(tsY,pred_y)
    return mean_7,mean_8,sd_7,sd_8,accuracy_7,accuracy_8,accuracy_total

