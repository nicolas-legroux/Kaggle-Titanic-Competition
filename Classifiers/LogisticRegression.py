from sklearn.linear_model import LogisticRegression
import crossvalidation
import numpy as np
import readAndClean


def getLogisticRegressionPrediction(X_train, y_train, X_test, classifierNames=[], printResult=False, test_IDs=[]):
    classifier = LogisticRegression()
    #Perform cross validation
    
    print "\n********* START LOGISTIC REGRESSION *********"    
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)     
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END LOGISTIC REGRESSION *********\n"
    
    classifierNames = classifierNames + ["LogisticRegression"]
      
    #Keeping only dummies for the logistic regression
    X_train_filtered, featuresname = readAndClean.keepLabels(readAndClean.computeSecondaryFeatures(X_test, X_train, False), True)
    X_test_filtered, featuresname = readAndClean.keepLabels(readAndClean.computeSecondaryFeatures(X_test, X_train, True), True)
    
    X_train_filtered = X_train_filtered.values
    X_test_filtered = X_test_filtered.values
    
    classifier.fit(X_train_filtered, y_train)
    
    predicted = classifier.predict(X_test_filtered)

    if printResult:    
        result = np.column_stack((test_IDs, predicted)).astype(int)
        np.savetxt('predictedLogisticRegression.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for Logistic Regression."
      
    return predicted, classifier, classifierNames