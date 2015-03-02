from sklearn.ensemble import ExtraTreesClassifier
import crossvalidation
import numpy as np
import readAndClean

def getExtraTreesPrediction(X_train, y_train, X_test, classifierNames=[], printResult=False, test_IDs=[],
                            n_estimators = 10, max_depth=None, min_samples_leaf=1, usingPandas=True):
    classifier = ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    #Perform cross validation
    
    print "\n********* START EXTRA TREES *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier,usingPandas=usingPandas)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END EXTRA TREES *********\n"
    
    classifierNames = classifierNames + ["Extra Trees"]
    
    X_train_filtered = None
    X_test_filtered = None    
    
    if(usingPandas):
        X_train_filtered, X_test_filtered = readAndClean.computeSecondaryFeatures(X_train, X_test)
        
        X_train_filtered, featuresname = readAndClean.keepLabels(X_train_filtered)
        X_test_filtered, featuresname = readAndClean.keepLabels(X_test_filtered)
       
        X_train_filtered = X_train_filtered.values
        X_test_filtered = X_test_filtered.values
    else:
        X_train_filtered = X_train
        X_test_filtered = X_test
    
    classifier.fit(X_train_filtered, y_train)
    
    predicted = classifier.predict(X_test_filtered)

    if printResult:    
        result = np.column_stack((test_IDs, predicted)).astype(int)
        np.savetxt('predictedExtraTrees.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for Extra Trees."
      
    return predicted, classifier, classifierNames, classifier.predict(X_train_filtered)

