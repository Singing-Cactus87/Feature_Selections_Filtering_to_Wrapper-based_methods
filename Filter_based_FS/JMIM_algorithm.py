import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score, f1_score, precision_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import kagglehub


#데이터셋 탑재
path = kagglehub.dataset_download("shebrahimi/financial-distress")

print("Path to dataset files:", path)

train_dir = "/root/.cache/kagglehub/datasets/shebrahimi/financial-distress/versions/1"
os.listdir(train_dir)

dt = pd.read_csv(os.path.join(train_dir, os.listdir(train_dir)[0]))
dt.head()

dt['Distress'] = np.array([0 if dt['Financial Distress'][i] > -0.5 else 1 for i in range(dt.shape[0])])
dt.drop(['Financial Distress'], axis=1, inplace=True)
dt.drop(['Company','Time'], axis=1, inplace=True)

X = dt.iloc[:,0:83]
y = dt.iloc[:,-1]

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.3,shuffle=True, random_state=321)
print(X_tr.shape, np.bincount(y_tr), np.bincount(y_te))



#연속형 데이터 이산형 자료로 변환

from sklearn.preprocessing import KBinsDiscretizer

disc = KBinsDiscretizer(n_bins=3,encode="ordinal",strategy="kmeans",random_state=321)


X_tr2 = disc.fit_transform(X_tr)
X_tr2 = pd.DataFrame(X_tr2, columns=X_tr.columns)


#JMIM Algorithm

def JMIM_filter(x,y,K):
  F = x.columns; S = []; t=0;JMIM = []
  vals =[]
  for i in range(len(F)):
    vals.append(mutual_info_classif(x.loc[:,F[i]].values.reshape(-1,1),y.values,n_neighbors=5,random_state=321))
  best_idx = np.argmax(vals); JMIM.append(np.max(vals))
  S.append(F[best_idx]); F = F.drop(F[best_idx]); t+=1
  while t < K:
    vals =[]
    for i in range(len(F)):
      cand = []
      for j in range(len(S)):
        ind = mutual_info_classif(x.loc[:,S[j]].values.reshape(-1,1),y.values,n_neighbors=5,random_state=321)
        prob_1 = float(np.bincount(x[S[j]])[0]/np.sum(np.bincount(x[S[j]])));x_1 = x[x[S[j]]==0];y_1 = y.values[x[S[j]]==0]
        prob_2 = float(np.bincount(x[S[j]])[1]/np.sum(np.bincount(x[S[j]])));x_2 = x[x[S[j]]==1];y_2 = y.values[x[S[j]]==1]
        prob_3 = float(np.bincount(x[S[j]])[2]/np.sum(np.bincount(x[S[j]])));x_3 = x[x[S[j]]==2];y_3 = y.values[x[S[j]]==2]; not_ind=0
        if ((prob_1 !=0) and (float(sum(np.bincount(x_1.loc[:,F[i]]))) > 5)):
          not_ind += prob_1*mutual_info_classif(x_1.loc[:,F[i]].values.reshape(-1,1),y_1,n_neighbors=5,random_state=321)
        if ((prob_2 !=0) and (float(sum(np.bincount(x_2.loc[:,F[i]]))) > 5)):
          not_ind += prob_2*mutual_info_classif(x_2.loc[:,F[i]].values.reshape(-1,1),y_2,n_neighbors=5,random_state=321)
        if ((prob_3 !=0) and (float(sum(np.bincount(x_3.loc[:,F[i]]))) > 5)):
          not_ind += prob_3*mutual_info_classif(x_3.loc[:,F[i]].values.reshape(-1,1),y_3,n_neighbors=5,random_state=321)
        cand.append(not_ind+ind)
      vals.append(np.min(cand))
    best_idx = np.argmax(vals); JMIM.append(np.max(vals))
    S.append(F[best_idx]); F = F.drop(F[best_idx]); t+=1
    if t%2==0: print(t)

  return S, JMIM


SET, result = JMIM_filter(X_tr2,y_tr,16)



#Macro F1-score 비교

X_tr_s = X_tr.loc[:,SET]
X_te_s = X_te.loc[:,SET]



rf1 = RandomForestClassifier(n_estimators=50, random_state=357)
rf1.fit(X_tr,y_tr)
print(f1_score(y_te,rf1.predict(X_te),average="macro"))

rf2 = RandomForestClassifier(n_estimators=50, random_state=357)
rf2.fit(X_tr_s,y_tr)
print(f1_score(y_te,rf2.predict(X_te_s),average="macro"))

from sklearn.model_selection import cross_validate
rf1 = RandomForestClassifier(n_estimators=50, random_state=357)
np.mean(cross_validate(rf1,X_tr,y_tr,cv=3, scoring=['f1_macro'])['test_f1_macro'])

from sklearn.model_selection import cross_validate
rf2 = RandomForestClassifier(n_estimators=50, random_state=357)
np.mean(cross_validate(rf2,X_tr_s,y_tr,cv=3, scoring=['f1_macro'])['test_f1_macro'])
