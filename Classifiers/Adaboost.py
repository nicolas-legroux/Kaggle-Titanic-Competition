from sklearn.ensemble import AdaBoostClassifier
import crossvalidation
import numpy as np


def getAdaboostPrediction(X_train, y_train, X_test, classifierNames=[], printResult=False, test_IDs=[]):
    classifier = AdaBoostClassifier()
    #Perform cross validation
        
    print "\n********* START ADABOOST *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END ADABOOST *********\n"
    
    classifierNames = classifierNames + ["Adaboost"]
    
    classifier.fit(X_train, y_train)
    
    predicted = classifier.predict(X_test)
    
    if printResult:    
        result = np.column_stack((test_IDs, predicted)).astype(int)
        np.savetxt('predictedAdaBoost.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for AdaBoost."
      
    return predicted, classifier, classifierNames   
