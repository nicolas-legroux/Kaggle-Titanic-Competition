# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 18:48:30 2015

@author: nicolas
"""

import numpy as np
import readAndClean
from Classifiers import Adaboost
from Classifiers import DecisionTree
from Classifiers import ExtraTrees
from Classifiers import GradientBoost
from Classifiers import LogisticRegression
from Classifiers import RandomForest

#Get Data
train_data, test_data, featureNames, data = readAndClean.getData()

train_IDs = train_data[:, 0]
test_IDs = test_data[:, 0]

#Get rid of Passenger IDs
train_data = np.array(train_data[:, 1:])
X_test = np.array(test_data[:, 1:])

X_train = train_data[:, 1::]
y_train = train_data[:, 0]

featureNames = featureNames[2:]

secondPass = False

#Get results of some training algorithms
RandomForest.getRandomForestPrediction(X_train, y_train, X_test, printResult=True, test_IDs=test_IDs)
DecisionTree.getDecisionTreePrediction(X_train, y_train, X_test, "DecisionTree", featureNames, printResult=True, test_IDs=test_IDs)
ExtraTrees.getExtraTreesPrediction(X_train, y_train, X_test, printResult=True, test_IDs=test_IDs)
GradientBoost.getGradientBoostPrediction(X_train, y_train, X_test, printResult=True, test_IDs=test_IDs)
LogisticRegression.getLogisticRegressionPrediction(X_train, y_train, X_test, printResult=True, test_IDs=test_IDs)
Adaboost.getAdaboostPrediction(X_train, y_train, X_test, printResult=True, test_IDs=test_IDs)

if secondPass:

    classifierNames = []
    
    predictedForestTrain, classifierForest, classifierNames =  \
    RandomForest.getRandomForestPrediction(X_train, y_train, X_train, classifierNames)
    
    predictedTreeTrain, classifierTree, classifierNames =  \
    DecisionTree.getDecisionTreePrediction(X_train, y_train, X_train, "FirstTree", featureNames)
    
    predictedExtraTrain, classifierExtra, classifierNames = \
    ExtraTrees.getExtraTreesPrediction(X_train, y_train, X_train, classifierNames)
    
    predictedGradientTrain, classifierGradient, classifierNames = \
    GradientBoost.getGradientBoostPrediction(X_train, y_train, X_train, classifierNames)
    
    predictedLogisticTrain, classifierLogistic, classifierNames = \
    LogisticRegression.getLogisticRegressionPrediction(X_train, y_train, X_train, classifierNames)
    
    predictedAdaboostTrain, classifierAdaboost, classifierNames = \
    Adaboost.getAdaboostPrediction(X_train, y_train, X_train, classifierNames)
    
    X_2ndpass_train = np.array((predictedForestTrain,
                         predictedTreeTrain,
                         predictedExtraTrain,
                         predictedGradientTrain,
                         predictedLogisticTrain,
                         predictedAdaboostTrain)
                         )
    X_2ndpass_train = X_2ndpass_train.transpose()
    
    X_2ndpass_test = np.array((classifierForest.predict(X_test),
                                classifierTree.predict(X_test),
                                classifierExtra.predict(X_test),
                                classifierGradient.predict(X_test),
                                classifierLogistic.predict(X_test),
                                classifierAdaboost.predict(X_test))
                         )
                         
    X_2ndpass_test = X_2ndpass_test.transpose()
    
    secondpassclassifiersname = []
    
    predicted,classifierfinal, whatever = DecisionTree.getDecisionTreePrediction(X_2ndpass_train, y_train, X_2ndpass_test, "FinalTree", classifierNames)
        
    result = np.column_stack((test_IDs, predicted)).astype(int)
    
    np.savetxt('predicted.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
    print "File written"


