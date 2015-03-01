from sklearn.linear_model import LogisticRegression
import crossvalidation
import numpy as np


def getLogisticRegressionPrediction(X_train, y_train, X_test, classifierNames=[], printResult=False, test_IDs=[]):
    classifier = LogisticRegression()
    #Perform cross validation
    
    print "\n********* START LOGISTIC REGRESSION *********"    
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)     
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END LOGISTIC REGRESSION *********\n"
    
    classifierNames = classifierNames + ["LogisticRegression"]
     
    classifier.fit(X_train, y_train)
    
    predicted = classifier.predict(X_test)

    if printResult:    
        result = np.column_stack((test_IDs, predicted)).astype(int)
        np.savetxt('predictedLogisitcRegression.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for Logisitic Regression."
      
    return predicted, classifier, classifierNames