from sklearn.ensemble import ExtraTreesClassifier
import crossvalidation


def getExtraTreesPrediction(X_train, y_train, X_test, featuresname, classifiersname):
    classifier = ExtraTreesClassifier(n_estimators=2, min_samples_leaf = 25)
    #Perform cross validation
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)
    
    print "********* START EXTRA FOREST *********"
    print "Training error" , errortraining 
    print "Test error : " , errortest
    print "********* END EXTRA FOREST *********"
    
    classifiersname = classifiersname + ["Extra Trees"]
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test),classifier, classifiersname