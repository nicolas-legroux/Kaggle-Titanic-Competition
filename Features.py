# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


def showFeaturesImportance(X, y, featuresname):
    featuresname = np.array(featuresname)
    
    print "Starting evaluating features importance..."    
    
    classifier = RandomForestClassifier(oob_score=True, n_estimators=10000)

    classifier = classifier.fit(X,y)
    
   
    feature_importance = classifier.feature_importances_

    feature_importance = 100.0 * (feature_importance / feature_importance.max())
      
    
    
    fi_threshold = 15
    important_idx = np.where(feature_importance > fi_threshold)[0]
    important_features = featuresname[important_idx]
     
    
    print "\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance):\n", \
            important_features
     
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    print "\nFeatures sorted by importance (DESC):\n", important_features[sorted_idx]
     
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.draw()
    plt.show()
    
    print "Evaluating features importance done."    