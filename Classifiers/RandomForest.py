from sklearn.ensemble import RandomForestClassifier
import crossvalidation
import numpy as np


def getRandomForestPrediction(X_train, y_train, X_test, classifierNames=[], printResult=False, test_IDs=[]):
    classifier = RandomForestClassifier(n_estimators=2228, min_samples_split = 7, min_samples_leaf=2, criterion='entropy', max_features=6)
    #Perform cross validation
    
    print "\n********* START RANDOM FOREST *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END RANDOM FOREST *********\n"
    
    classifierNames = classifierNames + ["RandomForest"]
    
    classifier.fit(X_train, y_train)
    
    predicted = classifier.predict(X_test)

    if printResult:    
        result = np.column_stack((test_IDs, predicted)).astype(int)
        np.savetxt('predictedRandomForest.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for Random Forest."
      
    return predicted, classifier, classifierNames