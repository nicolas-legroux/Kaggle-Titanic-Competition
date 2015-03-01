from sklearn.linear_model import LogisticRegression
import crossvalidation


def getLogisticRegressionPrediction(X_train, y_train, X_test, featuresname, classifiersname):
    classifier = LogisticRegression()
    #Perform cross validation
    
    print "\n********* START LOGISTIC REGRESSION *********"    
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)     
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END LOGISTIC REGRESSION *********\n"
    
    classifiersname = classifiersname + ["LogisticRegression"]
     
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test),classifier, classifiersname