"""
CSDS340 Case Study 1

@author: Clay Preusch, Grant Konkel
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

seed = 42

# random forest 
rf = RandomForestClassifier(
    n_estimators = 1500,
    criterion = 'log_loss',
    max_depth = 10,
    min_samples_split = 2,
    min_samples_leaf = 1,
    max_features = 'log2',
    bootstrap = False,
    random_state = seed
)

def aucCV(features, labels):

    # select using SelectFromModel, which selects features based on a RandomForest's feature weights
    selector = SelectFromModel(rf, threshold='median')

    # make model using pipeline
    model = make_pipeline(selector, rf)

    # predict cross validation scores
    scores = cross_val_score(model, features, labels, cv=10, scoring='roc_auc')
    return scores

def predictTest(trainFeatures, trainLabels, testFeatures):

    # select using SelectFromModel, which selects features based on a RandomForest's feature weights
    selector = SelectFromModel(rf, threshold='median')

    # make and fit model using pipelines
    model = make_pipeline(selector, rf)
    model.fit(trainFeatures, trainLabels)

    # predict test outputs
    testOutputs = model.predict_proba(testFeatures)[:, 1]
    return testOutputs

if __name__ == "__main__":

    # load data
    spamTrain1 = np.loadtxt('spamTrain1.csv',delimiter=',')
    spamTrain2 = np.loadtxt('spamTrain2.csv',delimiter=',')
    combinedData = np.vstack((spamTrain1, spamTrain2))

    # shuffle data
    shuffleIndex = np.arange(np.shape(combinedData)[0])
    np.random.shuffle(shuffleIndex)
    combinedData = combinedData[shuffleIndex, :]

    # seperate features & labels
    features = combinedData[:,:-1]
    labels = combinedData[:,-1]

    print("10-fold cross-validation mean AUC: ", np.mean(aucCV(features,labels)))

    trainFeatures = features[0::2,:]
    trainLabels = labels[0::2]
    testFeatures = features[1::2,:]
    testLabels = labels[1::2]
    testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)

    print("Test set AUC: ", roc_auc_score(testLabels,testOutputs))
    
    # calculations for graphing feature importances
    rf.fit(trainFeatures, trainLabels)
    importances = rf.feature_importances_
    selector = SelectFromModel(rf, threshold='median')
    model = make_pipeline(selector, rf)
    model.fit(trainFeatures, trainLabels)
    median_importance = np.median(importances)

    # plot feature importances
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances, align="center")
    plt.xticks(range(len(importances)), range(len(importances)), rotation=90)
    plt.axhline(y=median_importance, color='r', linestyle='--', label="Median Importance")
    plt.ylabel('Importance Value')
    plt.xlabel('Features')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot outputs vs labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    plt.subplot(2,1,1)
    plt.plot(np.arange(nTestExamples),testLabels[sortIndex],'b.')
    plt.ylabel('Target')
    plt.subplot(2,1,2)
    plt.plot(np.arange(nTestExamples),testOutputs[sortIndex],'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')
    plt.tight_layout()
    plt.show()
