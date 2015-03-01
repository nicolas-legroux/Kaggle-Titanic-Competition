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
train_data, test_data, featuresname, data = readAndClean.getData()

#Get rid of Passenger IDs
train_data = np.array(train_data[:, 1:])
test_data_NoID = np.array(test_data[:, 1:])

X_train = train_data[:, 1::]
y_train = train_data[:, 0]

featuresname = featuresname[2:]

X_test =  test_data_NoID

#Go with a few classifier

#predictedForest = RandomForest.getRandomForestPrediction(X_train, y_train, X_test)
#predictedTree = DecisionTree.getDecisionTreePrediction(X_train, y_train, X_test)
#predictedExtra = ExtraTrees.getExtraTreesPrediction(X_train, y_train, X_test)
#predictedGradient = GradientBoost.getGradientBoostPrediction(X_train, y_train, X_test)
#predictedLogistic = LogisticRegression.getLogisticRegressionPrediction(X_train, y_train, X_test)
#predictedAdaboost = Adaboost.getAdaboostPrediction(X_train, y_train, X_test)

classifiersname = []

predictedForestTrain,classifierForest,classifiersname = RandomForest.getRandomForestPrediction(X_train, y_train, X_train,featuresname,classifiersname)
predictedTreeTrain,classifierTree,classifiersname = DecisionTree.getDecisionTreePrediction(X_train, y_train, X_train,featuresname,classifiersname,"firstTree")
predictedExtraTrain,classifierExtra,classifiersname = ExtraTrees.getExtraTreesPrediction(X_train, y_train, X_train,featuresname,classifiersname)
predictedGradientTrain,classifierGradient,classifiersname = GradientBoost.getGradientBoostPrediction(X_train, y_train, X_train,featuresname,classifiersname)
predictedLogisticTrain,classifierLogistic,classifiersname = LogisticRegression.getLogisticRegressionPrediction(X_train, y_train, X_train,featuresname,classifiersname)
predictedAdaboostTrain,classifierAdaboost,classifiersname = Adaboost.getAdaboostPrediction(X_train, y_train, X_train,featuresname,classifiersname)



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

predicted,classifierfinal,secondpassclassifiersname = DecisionTree.getDecisionTreePrediction(X_2ndpass_train, y_train, X_2ndpass_test,classifiersname,secondpassclassifiersname,"finalTree")




IDs = test_data[:, 0]

result = np.column_stack((IDs, predicted)).astype(int)

#Features.showFeaturesImportance(train_data[:, 1::], train_data[:,0], labels[2:])
#parametersOptimisation.fitParameters(train_data[:, 1::], train_data[:,0])
#learningCurve.drawLearningCurve(train_data[:, 1::], train_data[:,0])

np.savetxt('predicted.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
print "File written"

"""
dot_data = StringIO.StringIO()
tree.export_graphviz(classifier, out_file=dot_data, feature_names=labels[2:])
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanic.png')
"""


