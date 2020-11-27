import os
import config
import model_dispatcher
import argparse 

import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree 
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


kfold = StratifiedKFold(n_splits= config.FOLDS) #must be equal to FOLDS

# X,y = make_classification(n_samples= 100, n_features= 25)


# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

h_file = pd.read_csv(config.HYPER_FILE)
y = h_file.pop(h_file.iloc[:, config.TARGET].name)
X = h_file


RFC = tree.DecisionTreeClassifier(criterion= "gini", random_state = 42)
gsRFC = GridSearchCV(RFC, param_grid = model_dispatcher.DTG_PARAMS, cv = kfold, scoring = "accuracy",
                    n_jobs = 1, verbose = 1)

gsRFC.fit(X, y)
    
RFC_best = gsRFC.best_estimator_

#Best score
print("Best score:", gsRFC.best_score_)
print("Best parameters:", RFC_best)

