#Testing loading models and predict
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

#Algorithms to work with
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

import glob #For importing files


# mods = {"gsRFC": gsRFC, "gsGBC":gsGBC, "gsExtC":gsExtC, "gsLDA":gsLDA}
# # mods = { "GradientBoosting":gGB, "Random Forest":gRF, "Extra Trees": gET,
#         "AdaBoost": gADA, "SVC": grid}


def val_scores(all_files, X_val, y_val):
    scores =pd.DataFrame(columns = ["Model", "Accuracy", "F1_Score"])
  
    for filename in all_files:
        model = joblib.load(filename)
        result = model.score(X_val, y_val)
        scores = scores.append({"Model":filename, "Accuracy":result,
                               "F1_Score":metrics.f1_score(model.predict(X_val), y_val)},
                               ignore_index=True )
    print(scores.sort_values(by = "Accuracy", ascending = False))
    return scores




if __name__ == "__main__":
    target = config.TARGET
    num_folds = config.FOLDS
    seed = config.SEED
    X_train =  pd.read_csv("../input/X_train.csv", index_col=0)
    y_train =  pd.read_csv("../input/y_train.csv").values.ravel()
    X_val =  pd.read_csv("../input/X_val.csv", index_col=0)
    y_val =  pd.read_csv("../input/y_val.csv").values.ravel()

    scoring = config.SCORING
    all_files = glob.glob("../models" + "/*.bin")

    val_scores(all_files, X_val, y_val)
    
