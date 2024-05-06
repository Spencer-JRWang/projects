import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

df_all = pd.read_csv('data/data.txt', sep='\t')
from itertools import combinations
category = ["A", "B", "C", "D"]
category = list(combinations(category, 2))
print(category)

from model.rfe import model_rfe
from model.model import grid_search
from model.model import stacking_model
f = open("Outcome/Feature_Selection.txt", "w")
for i in category:
    Cat_A = i[0]
    Cat_B = i[1]
    print(f"---------------{Cat_A} vs {Cat_B}---------------")
    print("Best Feature Combination Detecting...", end = " ")
    f.write("--------------------" + str(Cat_A) + " and " + str(Cat_B) + "--------------------\n")
    core = LGBMClassifier(n_estimators=1000, verbose = -1, max_depth = 5)
    df = df_all[df_all['Disease'].isin([Cat_A, Cat_B])]
    best = model_rfe(f, core, df, Cat_A, Cat_B)
    print("Success")

    all_com = grid_search()
    AUCs = []
    Scores = []
    print("Stacking model is building...")
    for m in tqdm(all_com):
        IntegratedScore = stacking_model(df[best], df['Disease'].map({Cat_A: 0, Cat_B: 1}),list(m))
        Scores.append(IntegratedScore)
        fpr, tpr, thresholds = roc_curve(IntegratedScore.iloc[:, 0], IntegratedScore.iloc[:, 1])
        roc_auc = auc(fpr, tpr)
        f.write("Stacking Model: " + str([i[0] for i in m]) + "\n")
        f.write("ROC_AUC: " + str(roc_auc) + "\n")
        AUCs.append(roc_auc)
    print("Success")
    best_stacking = []
    for t in all_com[AUCs.index(max(AUCs))]:
        best_stacking.append(t[0])
    f.write("Best Stacking Model detected " + str(best_stacking) + "\n")
    f.write("Best IntegratedScore AUC = " + str(max(AUCs)) + "\n")
    print("Best Stacking Model detected " + str(best_stacking))
    print("Best IntegratedScore AUC = " + str(max(AUCs)))
f.close()

