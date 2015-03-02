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
X_train, y_train, X_test, test_IDs = readAndClean.getData()

test_IDs = test_IDs.values
y_train = y_train.values

#parametersOptimisation.fitParameters(X_train,y_train)

secondPass = True

#Get results of some training algorithms
RandomForest.getRandomForestPrediction(X_train, y_train, X_test, printResult=True, test_IDs=test_IDs, n_estimators=30, min_samples_leaf=20)
DecisionTree.getDecisionTreePrediction(X_train, y_train, X_test, "DecisionTree", printResult=True, test_IDs=test_IDs, max_depth=2)
ExtraTrees.getExtraTreesPrediction(X_train, y_train, X_test, printResult=True, test_IDs=test_IDs, n_estimators=30, min_samples_leaf=20)
GradientBoost.getGradientBoostPrediction(X_train, y_train, X_test, printResult=True, test_IDs=test_IDs)
LogisticRegression.getLogisticRegressionPrediction(X_train, y_train, X_test, printResult=True, test_IDs=test_IDs)
Adaboost.getAdaboostPrediction(X_train, y_train, X_test, printResult=True, test_IDs=test_IDs)

if secondPass:

    classifierNames = []
    
    predictedForestTest, classifierForest, classifierNames, predictedForestTrain =  \
    RandomForest.getRandomForestPrediction(X_train, y_train, X_test, classifierNames=classifierNames, printResult=True, test_IDs=test_IDs, n_estimators=30, min_samples_leaf=20)
    
    predictedExtraTest, classifierExtra, classifierNames, predictedExtraTrain =  \
    RandomForest.getRandomForestPrediction(X_train, y_train, X_test, classifierNames=classifierNames, printResult=True, test_IDs=test_IDs, n_estimators=30, min_samples_leaf=20)
    
    
    predictedTreeTest, classifierTree, classifierNames, predictedTreeTrain =  \
    DecisionTree.getDecisionTreePrediction(X_train, y_train, X_test, "DecisionTreePass1", classifierNames=classifierNames, printResult=True, test_IDs=test_IDs, max_depth=2)
        
    predictedLogisticTest, classifierLogistic, classifierNames, predictedLogisticTrain = \
    LogisticRegression.getLogisticRegressionPrediction(X_train, y_train, X_test, classifierNames=classifierNames)
    
    X_2ndpass_train = np.array((predictedForestTrain,
                         predictedTreeTrain,                        
                         predictedLogisticTrain,
                        )
                         )
    X_2ndpass_train = X_2ndpass_train.transpose()
    
    X_2ndpass_test = np.array((predictedForestTest,
                                predictedTreeTest,                                                             
                                predictedLogisticTest,
                               )
                         )
                         
    X_2ndpass_test = X_2ndpass_test.transpose()
    
    predicted_sum = predictedForestTest + predictedTreeTest + predictedLogisticTest
    
    predicted_voting = np.where((predicted_sum >= 2), 1, 0)
    
    secondpassclassifiersname = []
    
    predicted, classifierfinal, whatever, w = DecisionTree.getDecisionTreePrediction(
        X_2ndpass_train, y_train, X_2ndpass_test, 'DecisionTreePass2', featuresname=classifierNames, usingPandas=False, max_depth=5)
        
    result = np.column_stack((test_IDs, predicted)).astype(int)
    
    np.savetxt('predictedBagging.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
    print "File written"


