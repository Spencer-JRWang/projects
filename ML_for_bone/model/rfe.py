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


def model_rfe(f,core,df,cat_A,cat_B):
    X = df.drop("Disease",axis = 1)
    y = df['Disease']
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    shuffle_index = np.random.permutation(X.index)
    X = X.iloc[shuffle_index]
    y = y.iloc[shuffle_index]
    y_encode = y.map({cat_A: 0, cat_B: 1})
    outcome_feature = []
    outcome_score = []
    for i in range(X.shape[1]):
        rfe = RFE(core, n_features_to_select=i + 1)
        rfe.fit(X, y_encode)
        selected_features = X.columns[rfe.support_]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(core, X[selected_features], y_encode, cv=cv)
        selected_features = X.columns[rfe.support_]
        outcome_feature.append(selected_features)
        outcome_score.append(scores.mean())
    
    # 数据层面的最佳组合
    max_predict_data = max(outcome_score)
    best_features = list(outcome_feature[outcome_score.index(max_predict_data)])
    f.write("Best Features Combination Detected: " + str(best_features) + "\n")
    f.write("Best Validation Score: " + str(max_predict_data) + "\n")

    return best_features

 