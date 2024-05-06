import warnings
warnings.filterwarnings('ignore')
import os
import shutil
import pandas as pd
import numpy as np
from sklearn import svm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def grid_search():
    base_model = [
        #('RandomForest',RandomForestClassifier(n_estimators=2500)),
        #('GradientBoost',GradientBoostingClassifier(n_estimators=1000)),
        ('LGBM',LGBMClassifier(verbose = -1,n_estimators = 1000, max_depth = 5)),
        ('XGBoost',XGBClassifier(n_estimators = 1000, max_depth = 5)),
        ('CatBoost',CatBoostClassifier(verbose = False,iterations = 800, max_depth = 5))
    ]
    from itertools import combinations
    all_combinations = []
    for r in range(1, len(base_model) + 1):
        combinations_r = combinations(base_model, r)
        all_combinations.extend(combinations_r)
    return all_combinations

def stacking_model(X,y_encode,base_model):
    scores_st = []
    X = X.reset_index(drop=True)
    y_encode = y_encode.reset_index(drop=True)
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    shuffle_index = np.random.permutation(X.index)
    X = X.iloc[shuffle_index]
    y_encode = y_encode.iloc[shuffle_index]
    meta_model = LogisticRegression(max_iter=10000000)
    stacking_clf = StackingClassifier(estimators=base_model, final_estimator=meta_model, stack_method='predict_proba')
    score_st = cross_val_predict(stacking_clf, X, y_encode, cv=stratified_kfold, method="predict_proba")
    scores_st.append(score_st[:, 1])
    scores_st = np.array(scores_st)
    scores_st = np.mean(scores_st, axis=0)
    dff = y_encode.to_frame()
    dff["IntegratedScore"] = scores_st
    return dff
