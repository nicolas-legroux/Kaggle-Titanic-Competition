from sklearn.ensemble import RandomForestClassifier
import crossvalidation
import numpy as np
import readAndClean


def getRandomForestPrediction(X_train, y_train, X_test, classifierNames=[], printResult=False, test_IDs=[]):
    classifier = RandomForestClassifier(n_estimators=1000, min_samples_leaf=20)
    #Perform cross validation
    
    print "\n********* START RANDOM FOREST *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END RANDOM FOREST *********\n"
    
    classifierNames = classifierNames + ["RandomForest"]
    
    X_train_filtered, featuresname = readAndClean.keepLabels(readAndClean.computeSecondaryFeatures(X_test, X_train, False))
    X_test_filtered, featuresname = readAndClean.keepLabels(readAndClean.computeSecondaryFeatures(X_test, X_train, True))
   
    X_train_filtered = X_train_filtered.values
    X_test_filtered = X_test_filtered.values
    
    classifier.fit(X_train_filtered, y_train)
    
    predicted = classifier.predict(X_test_filtered)

    if printResult:    
        result = np.column_stack((test_IDs, predicted)).astype(int)
        np.savetxt('predictedRandomForest.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for Random Forest."
      
    return predicted, classifier, classifierNames