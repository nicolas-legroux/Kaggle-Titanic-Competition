import crossvalidation
from sklearn import tree
import StringIO
import pydot


def getDecisionTreePrediction(X_train, y_train, X_test, featuresname, classifiersname, treefilename):
    classifier = tree.DecisionTreeClassifier()
    
    errortraining, errortest = crossvalidation.crossValidation(X_train, y_train, classifier)

    print "********* START DECISION TREE *********"
    print "Training error" , errortraining 
    print "Test error : " , errortest
    print "********* END DECISION TREE *********"
    classifiersname = classifiersname + ["Decision Tree"]
    classifier.fit(X_train, y_train)
    
    dot_data = StringIO.StringIO()
    tree.export_graphviz(classifier, out_file=dot_data, feature_names=featuresname)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(treefilename + '.png')
    
    
    return classifier.predict(X_test),classifier, classifiersname