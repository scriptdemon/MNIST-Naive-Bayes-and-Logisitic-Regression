# -*- coding: utf-8 -*-
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt

def calculateDerivative(weightMatrix,xMatrix,yVector):
    derivative = np.zeros(len(xMatrix[0]),)
    constant_term = np.subtract(yVector,logitValue(weightMatrix,xMatrix))
    derivative = np.dot(constant_term,xMatrix)
    return derivative

def extractFeatures(trX):
    return list(zip(getMeanTestFeatures(trX),getSdTestFeatures(trX)))

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

def gradientAscentOptimize(txSet,tY,lRate,itr):
    new_txSet = []
    for i in txSet:
        l = [1]
        for j in i:
            l.append(j)
        new_txSet.append(l)
    new_txSet = np.array(new_txSet)
    weights = np.zeros((len(new_txSet[0]),))
    iteration = []
    plot = []
    for i in range(itr):
        derivative = calculateDerivative(weights,new_txSet,tY)
        derivative_with_learning = np.dot(derivative,lRate)
        weights = np.add(weights,derivative_with_learning)
        logLikelihood = getLogLikelihood(new_txSet,tY,weights)
        if(i%10000 == 0 and i>0):
            iteration.append(i)
            plot.append(logLikelihood)
    return weights,iteration,plot

def logitValue(weight,x):
    sum_of_products = np.dot(weight,np.transpose(x))
    return 1.0/(1 + np.exp(-sum_of_products))

def testLogisticRegression(txSet,weights):
    new_txSet = []
    for i in txSet:
        l = [1]
        for j in i:
            l.append(j)
        new_txSet.append(l)
    new_txSet = np.array(new_txSet)
    pred_y = []
    for i in new_txSet:
        pred = logitValue(weights,i)
        if(pred >= 0.5):
            pred_y.append(1)
        else:
            pred_y.append(0)
    return pred_y

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
    return hits_7/total_7,hits_8/total_8,(hits_7 + hits_8)/(total_7+total_8)

def getLogLikelihood(txSet,tY,weights):
    a = np.dot(tY,txSet)
    b = np.dot(weights,np.transpose(a))
    c = np.dot(weights,np.transpose(txSet))
    d = np.sum(np.log(1+np.exp(c)))
    return b -d

def plotLikelihood(itr,plot):
    plt.plot(itr,plot)
    plt.xlabel("Iterations")
    plt.ylabel("Likelihood")
    plt.title("Change in log likelihood")
    
def logisitcRegression(trX,trY,tsX,tsY):
    w,itr,plot = gradientAscentOptimize(trX,trY,1e-3,200000)
    pred_y = testLogisticRegression(tsX,w)
    accuracy_7,accuracy_8,accuracy_total = getAccuracy(tsY,pred_y)
    plotLikelihood(itr,plot)
    return w,accuracy_7,accuracy_8,accuracy_total