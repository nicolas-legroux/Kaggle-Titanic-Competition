from sklearn.linear_model import LogisticRegression
import crossvalidation


def getLogisticRegressionPrediction(X_train, y_train, X_test, featuresname, classifiersname):
    classifier = LogisticRegression()
    #Perform cross validation
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)
    
    print "********* START LOGISTIC REGRESSION *********"
    print "Training error" , errortraining 
    print "Test error : " , errortest
    print "********* END LOGISTIC REGRESSION *********"
    
    classifiersname = classifiersname + ["LogisticRegression"]
     
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test),classifier, classifiersname