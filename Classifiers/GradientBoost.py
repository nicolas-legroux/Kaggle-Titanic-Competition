from sklearn.ensemble import GradientBoostingClassifier
import crossvalidation
import numpy as np

def getGradientBoostPrediction(X_train, y_train, X_test, classifierNames=[], printResult=False, test_IDs=[]):
    classifier = GradientBoostingClassifier()
    #Perform cross validation
    print "\n********* START GRADIENT BOOST *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END GRADIENT BOOST *********\n"
    
    classifierNames = classifierNames + ["GradientBoost"]
    classifier.fit(X_train, y_train)

    predicted = classifier.predict(X_test)

    if printResult:    
        result = np.column_stack((test_IDs, predicted)).astype(int)
        np.savetxt('predictedGradientBoost.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for GradientBoost."
      
    return predicted, classifier, classifierNames