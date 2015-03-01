# -*- coding: utf-8 -*-
import random
import numpy as np

def crossValidation(X,y,classifier):
    #Perform Cross Validation
    #Set Cross Validation parameters
    random.seed()
    numberOfLines = X.shape[0]
    
    r = [i for i in range(numberOfLines)]
    random.shuffle(r)

    cross_validation_folds = 10
    num = float(len(r))/cross_validation_folds
    cross_validation_chunks = [ r [i:i + int(num)] for i in range(0, (cross_validation_folds -1)*int(num), int(num))]
    cross_validation_chunks.append(r[(cross_validation_folds -1)*int(num):])

    resultsTrainingSet = []
    resultsTestSet = []
    
    print "Starting Cross-Validation..."
    for i in range(cross_validation_folds):
        print "   Done with Pass " + str(i)
        test_idx = cross_validation_chunks[i]
        train_idx = [x for x in r if x not in test_idx]
        train_data_cross = X[train_idx, :]
        test_data_cross = X[test_idx, :]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        classifier = classifier.fit(train_data_cross, y_train)
        resultsTrainingSet.append(classifier.score(train_data_cross, y_train))
        resultsTestSet.append(classifier.score(test_data_cross, y_test))
        
    print "Done with Cross-Validation"
    return np.array(resultsTrainingSet).mean(), np.array(resultsTestSet).mean()