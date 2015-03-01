from sklearn.ensemble import RandomForestClassifier
import crossvalidation


def getRandomForestPrediction(X_train, y_train, X_test, featuresname,classifiersname):
    classifier = RandomForestClassifier(n_estimators=2, min_samples_leaf = 25)
    #Perform cross validation
    
    print "\n********* START RANDOM FOREST *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END RANDOM FOREST *********\n"
    
    classifiersname = classifiersname + ["RandomForest"]
    
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test),classifier,classifiersname