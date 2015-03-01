from sklearn.ensemble import GradientBoostingClassifier
import crossvalidation


def getGradientBoostPrediction(X_train, y_train, X_test, featuresname, classifiersname):
    classifier = GradientBoostingClassifier()
    #Perform cross validation
    print "\n********* START GRADIENT BOOST *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END GRADIENT BOOST *********\n"
    
    classifiersname = classifiersname + ["GradientBoost"]
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test),classifier, classifiersname