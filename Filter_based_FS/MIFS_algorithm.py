import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score, f1_score, precision_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


#데이터셋 준비
import kagglehub

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



#MIFS 알고리즘 함수로 정의

def MIFS_filter(x,y,k,b=0.6):
  F = x.columns ; S = []; MIFS = []; val = []; t = 0
  for i in range(x.shape[1]):
    val.append(mutual_info_classif(x.iloc[:,i].values.reshape(-1,1),y.values,n_neighbors=5,random_state=321))
  best_idx = np.argmax(val); MIFS.append(np.max(val))
  S.append(F[best_idx]) ; F = F.drop(F[best_idx]); t += 1
  while t < k:
    val = []
    for j in range(len(F)):
      rel = mutual_info_classif(x.iloc[:,j].values.reshape(-1,1),y.values,n_neighbors=5,random_state=321)
      rd = 0
      for q in range(len(S)):
        rd += mutual_info_regression(x.iloc[:,j].values.reshape(-1,1), x.iloc[:,q].values,n_neighbors=5,random_state=321)
      val.append(rel-b*rd)
    best_idx = np.argmax(val); MIFS.append(np.max(val))
    S.append(F[best_idx]) ; F = F.drop(F[best_idx]); t += 1
    if (t%2==0): print(t)

  return S, MIFS


#MIFS 기반 변수선택

SET, Met = MIFS_filter(X_tr,y_tr,16,0.6)

X_tr_s = X_tr.loc[:,SET]
X_te_s = X_te.loc[:,SET]

#성능 비교(full dataset, reduced dataset)

#full
rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=321)
rf1.fit(X_tr,y_tr)
print(f1_score(y_te,rf1.predict(X_te),average="macro"))

#reduced
rf2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=321)
rf2.fit(X_tr_s,y_tr)
print(f1_score(y_te,rf2.predict(X_te_s),average="macro"))
