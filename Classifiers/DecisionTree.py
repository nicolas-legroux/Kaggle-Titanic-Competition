import crossvalidation
from sklearn import tree
import StringIO
import pydot


def getDecisionTreePrediction(X_train, y_train, X_test, featuresname, classifiersname, treefilename):
    classifier = tree.DecisionTreeClassifier()
    
    print "\n********* START DECISION TREE *********"
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)    
    print "Score on Training Set" , errortraining 
    print "Score on Test Set : " , errortest
    print "********* END DECISION TREE *********\n"
    
    classifiersname = classifiersname + ["Decision Tree"]
    classifier.fit(X_train, y_train)
    
    dot_data = StringIO.StringIO()
    tree.export_graphviz(classifier, out_file=dot_data, feature_names=featuresname)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(treefilename + '.png')
    
    
    return classifier.predict(X_test),classifier, classifiersname