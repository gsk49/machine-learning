# -*- coding: utf-8 -*-
"""
Script used to evaluate classifier accuracy

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
from classifySpam import predictTest

desiredFPR = 0.01
trainDataFilename = 'spamTrain1.csv'
testDataFilename = 'spamTrain2.csv'
#testDataFilename = 'spamTest.csv'

def tprAtFPR(labels,outputs,desiredFPR):
    fpr,tpr,thres = roc_curve(labels,outputs)
    # True positive rate for highest false positive rate < 0.01
    maxFprIndex = np.where(fpr<=desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex+1]
    # Find TPR at exactly desired FPR by linear interpolation
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex+1]
    tprAt = ((tprAbove-tprBelow)/(fprAbove-fprBelow)*(desiredFPR-fprBelow) 
             + tprBelow)
    return tprAt,fpr,tpr

data1 = np.loadtxt(trainDataFilename,delimiter=',')
data2 = np.loadtxt(testDataFilename,delimiter=',')

combinedData = np.vstack((data1, data2))

shuffleIndex = np.arange(np.shape(combinedData)[0])
np.random.shuffle(shuffleIndex)
combinedData = combinedData[shuffleIndex, :]

# Splitting into train and test features
split_idx = int(0.5 * len(combinedData))  # Assuming a 50-50 split, adjust accordingly
trainFeatures = combinedData[:split_idx, :-1]
trainLabels = combinedData[:split_idx, -1]
testFeatures = combinedData[split_idx:, :-1]
testLabels = combinedData[split_idx:, -1]

testOutputs = predictTest(trainFeatures, trainLabels, testFeatures)
aucTestRun = roc_auc_score(testLabels, testOutputs)
tprAtDesiredFPR, fpr, tpr = tprAtFPR(testLabels, testOutputs, desiredFPR)

plt.plot(fpr, tpr)

print(f'Test set AUC: {aucTestRun}')
print(f'TPR at FPR = {desiredFPR}: {tprAtDesiredFPR}')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for spam detector')
plt.show()
