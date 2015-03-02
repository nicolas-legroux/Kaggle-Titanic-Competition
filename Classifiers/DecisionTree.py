import crossvalidation
from sklearn import tree
import StringIO
import pydot
import numpy as np
import readAndClean


def getDecisionTreePrediction(X_train, y_train, X_test, treeFileName, classifierNames=[], printResult=False, 
                              test_IDs=[], max_depth=4, usingPandas=True, featuresname=None):
    
    classifier = tree.DecisionTreeClassifier(max_depth=max_depth)    
    
    print "\n********* START DECISION TREE *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier,usingPandas=usingPandas)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END DECISION TREE *********\n"
    
    classifierNames = classifierNames + ["Decision Tree"]
    
    X_train_filtered = None
    X_test_filtered = None    
    
    if(usingPandas):
        X_train_filtered, X_test_filtered = readAndClean.computeSecondaryFeatures(X_train, X_test)
        
        X_train_filtered, featuresname = readAndClean.keepLabels(X_train_filtered)
        X_test_filtered, featuresname = readAndClean.keepLabels(X_test_filtered)
       
        X_train_filtered = X_train_filtered.values
        X_test_filtered = X_test_filtered.values
    else:
        X_train_filtered = X_train
        X_test_filtered = X_test
    
    classifier.fit(X_train_filtered, y_train)
    
    predicted = classifier.predict(X_test_filtered)
    
    dot_data = StringIO.StringIO()
    tree.export_graphviz(classifier, out_file=dot_data, feature_names=featuresname)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(treeFileName + '.png')


    if printResult:    
        result = np.column_stack((test_IDs, predicted)).astype(int)
        np.savetxt('predictedDecisionTree.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for Decision Tree."
      
    return predicted, classifier, classifierNames, classifier.predict(X_train_filtered)