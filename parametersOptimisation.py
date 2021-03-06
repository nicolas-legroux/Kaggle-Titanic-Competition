import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
import operator

def report(grid_scores, n_top=5):
    params = None
    top_scores = sorted(grid_scores, key=operator.itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Parameters with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f} (std: {1:.4f})".format(
              score.mean_validation_score, np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
        
        if params == None:
            params = score.parameters
    
    return params
    
def fitParameters(X,y):
     
    # The most common value for the max number of features to look at in each split is sqrt(# of features)
    sqrtfeat = np.sqrt(X.shape[1])
     
  
    # Large randomized test using max_depth to control tree size (5000 possible combinations)
    random_test1 = { "n_estimators"      : np.rint(np.linspace(X.shape[0]*2, X.shape[0]*4, 5)).astype(int),
                     "criterion"         : ["gini", "entropy"],
                     "max_features"      : np.rint(np.linspace(sqrtfeat/2, sqrtfeat*2, 5)).astype(int),
                     "max_depth"         : np.rint(np.linspace(1, X.shape[1]/2, 10)),
                     "min_samples_split" : np.rint(np.linspace(2, X.shape[0]/50, 10)).astype(int) }
     
    # Large randomized test using min_samples_leaf and max_leaf_nodes to control tree size (50k combinations)
    random_test2 = { "n_estimators"      : np.rint(np.linspace(X.shape[0]*2, X.shape[0]*4, 5)).astype(int),
                     "criterion"         : ["gini", "entropy"],
                     "max_features"      : np.rint(np.linspace(sqrtfeat/2, sqrtfeat*2, 5)).astype(int),
                     "min_samples_split" : np.rint(np.linspace(2, X.shape[0]/50, 10)).astype(int),
                     "min_samples_leaf"  : np.rint(np.linspace(1, X.shape[0]/200, 10)).astype(int)}
     
    forest = RandomForestClassifier(oob_score=True)
    
    grid_search = RandomizedSearchCV(forest, random_test2, n_jobs=-1, cv=10, n_iter=100)
    grid_search.fit(X, y)
    best_params_from_rand_search2 = report(grid_search.grid_scores_)