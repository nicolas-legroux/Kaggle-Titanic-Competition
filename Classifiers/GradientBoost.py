from sklearn.ensemble import GradientBoostingClassifier
import crossvalidation


def getGradientBoostPrediction(X_train, y_train, X_test, featuresname, classifiersname):
    classifier = GradientBoostingClassifier()
    #Perform cross validation
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)
    
    print "********* START GRADIENT BOOST *********"
    print "Training error" , errortraining 
    print "Test error : " , errortest
    print "********* END GRADIENT BOOST *********"
    
    classifiersname = classifiersname + ["GradientBoost"]
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test),classifier, classifiersname