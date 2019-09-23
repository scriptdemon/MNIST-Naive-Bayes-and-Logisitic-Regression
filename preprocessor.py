# -*- coding: utf-8 -*-
import statistics as stat

def extractFeatures(trX):
    return list(zip(getMeanTestFeatures(trX),getSdTestFeatures(trX)))
    
def splitTestSetClasswise(txSet,tySet,label):
    featureSet = []
    count = 0
    for i in txSet:
        if tySet[count] == label:
            featureSet.append(i)
        count+=1
    return featureSet

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
