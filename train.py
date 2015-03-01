# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 18:48:30 2015

@author: nicolas
"""

import numpy as np
import readAndClean
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import Features
import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve
from sklearn import tree
import pydot
import parametersOptimisation
import matplotlib.pyplot as plt
import learningCurve

random.seed()

#Choose a classifie
classifier = RandomForestClassifier(n_estimators=9000, min_samples_leaf = 25)

#Get Data
train_data, test_data, labels, data = readAndClean.getData()

#Get rid of Passenger IDs
train_data = np.array(train_data[:, 1:])
test_data_NoID = np.array(test_data[:, 1:])
numberOfLines = train_data.shape[0]

r = [i for i in range(numberOfLines)]
random.shuffle(r)

#Set Cross Validation parameters
cross_validation_folds = 10
num = float(len(r))/cross_validation_folds
cross_validation_chunks = [ r [i:i + int(num)] for i in range(0, (cross_validation_folds -1)*int(num), int(num))]
cross_validation_chunks.append(r[(cross_validation_folds -1)*int(num):])

"""
#Perform Cross Validation
resultsTrainingSet = []
resultsTestSet = []
for i in range(cross_validation_folds):
    test_IDs = cross_validation_chunks[i]
    train_IDs = [x for x in r if x not in test_IDs]
    train_data_cross = train_data[train_IDs, :]
    test_data_cross = train_data[test_IDs, :]
    classifier = classifier.fit(train_data_cross[:,1:],train_data_cross[:,0])
    resultsTrainingSet.append(classifier.score(train_data_cross[:, 1:], train_data_cross[:, 0]))
    resultsTestSet.append(classifier.score(test_data_cross[:, 1:], test_data_cross[:, 0]))

print np.array(resultsTrainingSet).mean()
print np.array(resultsTestSet).mean()
"""

classifier = classifier.fit(train_data[:, 1::], train_data[:,0])
predicted = classifier.predict(test_data_NoID)
IDs = test_data[:, 0]

result = np.column_stack((IDs, predicted)).astype(int)

#Features.showFeaturesImportance(train_data[:, 1::], train_data[:,0], labels[2:])
#parametersOptimisation.fitParameters(train_data[:, 1::], train_data[:,0])
#learningCurve.drawLearningCurve(train_data[:, 1::], train_data[:,0])

np.savetxt('predicted.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')

"""
dot_data = StringIO.StringIO()
tree.export_graphviz(classifier, out_file=dot_data, feature_names=labels[2:])
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanic.png')
"""


