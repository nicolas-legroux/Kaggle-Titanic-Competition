from sklearn.ensemble import AdaBoostClassifier
import crossvalidation


def getAdaboostPrediction(X_train, y_train, X_test, featuresname, classifiersname):
    classifier = AdaBoostClassifier()
    #Perform cross validation
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)
    
    print "********* START ADABOOST *********"
    print "Training error" , errortraining 
    print "Test error : " , errortest
    print "********* END ADABOOST *********"
    classifiersname = classifiersname + ["Adaboost"]
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test),classifier, classifiersname