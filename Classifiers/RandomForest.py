from sklearn.ensemble import RandomForestClassifier
import crossvalidation
import numpy as np
import readAndClean


def getRandomForestPrediction(X_train, y_train, X_test, classifierNames=[], printResult=False, test_IDs=[],
                              n_estimators=10, max_features='auto', min_samples_leaf=1, usingPandas=True):
    classifier = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
    
    #Perform cross validation
    
    print "\n********* START RANDOM FOREST *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier, usingPandas=usingPandas)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END RANDOM FOREST *********\n"
    
    classifierNames = classifierNames + ["RandomForest"]
    
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
        np.savetxt('predictedRandomForest.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for Random Forest."
      
    return predicted, classifier, classifierNames, classifier.predict(X_train_filtered)