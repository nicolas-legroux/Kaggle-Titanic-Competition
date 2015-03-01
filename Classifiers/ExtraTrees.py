from sklearn.ensemble import ExtraTreesClassifier
import crossvalidation
import numpy as np


def getExtraTreesPrediction(X_train, y_train, X_test, classifierNames=[], printResult=False, test_IDs=[]):
    classifier = ExtraTreesClassifier(n_estimators=2, min_samples_leaf = 25)
    #Perform cross validation
    
    print "\n********* START EXTRA TREES *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END EXTRA TREES *********\n"
    
    classifierNames = classifierNames + ["Extra Trees"]
    
    classifier.fit(X_train, y_train)

    predicted = classifier.predict(X_test)

    if printResult:    
        result = np.column_stack((test_IDs, predicted)).astype(int)
        np.savetxt('predictedExtraTrees.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for Extra Trees."
      
    return predicted, classifier, classifierNames

