import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestClassifier

import numpy as np

def drawLearningCurve(X,y):
    forest = RandomForestClassifier(n_estimators=500, max_depth=3)
    
    train_sizes, train_scores, test_scores = learning_curve.learning_curve(
            forest, X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(.1, 1., 10), verbose=0)
     
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
     
    plt.figure()
    plt.title("RandomForestClassifier")
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim((0.6, 1.01))
    plt.gca().invert_yaxis()
    plt.grid()
     
    # Plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Test score")
     
    # Plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                     alpha=0.1, color="r")
     
    # Draw the plot and reset the y-axis
    plt.draw()
    plt.show()
    plt.gca().invert_yaxis()