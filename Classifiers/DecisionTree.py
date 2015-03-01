import crossvalidation
from sklearn import tree
import StringIO
import pydot
import numpy as np


def getDecisionTreePrediction(X_train, y_train, X_test, treeFileName, featureNames, classifierNames=[], printResult=False, test_IDs=[]):
    classifier = tree.DecisionTreeClassifier()
    
    print "\n********* START DECISION TREE *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END DECISION TREE *********\n"
    
    classifierNames = classifierNames + ["Decision Tree"]
   
    classifier.fit(X_train, y_train)
    
    dot_data = StringIO.StringIO()
    tree.export_graphviz(classifier, out_file=dot_data, feature_names=featureNames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(treeFileName + '.png')
    
    
    predicted = classifier.predict(X_test)

    if printResult:    
        result = np.column_stack((test_IDs, predicted)).astype(int)
        np.savetxt('predictedDecisionTree.csv', result, fmt='%i', comments='', header='PassengerId,Survived', delimiter=',')
        print "File written for Decision Tree."
      
    return predicted, classifier, classifierNames