from sklearn.ensemble import GradientBoostingClassifier
import crossvalidation
import numpy as np
import readAndClean

def getGradientBoostPrediction(X_train, y_train, X_test, classifierNames=[], printResult=False, test_IDs=[], usingPandas=True):
    classifier = GradientBoostingClassifier()
    #Perform cross validation
    print "\n********* START GRADIENT BOOST *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier, usingPandas=usingPandas)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END GRADIENT BOOST *********\n"
    
    classifierNames = classifierNames + ["GradientBoost"]
    
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
        np.savetxt('predictedGradientBoost.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for GradientBoost."
      
    return predicted, classifier, classifierNames, classifier.predict(X_train_filtered)