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




#mRMR 알고리즘 함수 정의

def mRMR_filter(x,y,K,method = "MIQ"):
  F = x.columns[:] ; S = [] ; t=0; mRMR = []
  values = []
  for i in range(len(F)):
    values.append(mutual_info_classif(x.iloc[:,i].values.reshape(-1,1),y.values,n_neighbors=5,random_state=321))
  best_idx = np.argmax(values)
  S.append(F[best_idx]); F.drop(F[best_idx]);mRMR.append(np.max(values)); t += 1
  while t < K:
    MIQ = [];MID = []
    for j in range(len(F)):
      S_ = S[:]; S_.append(F[j])
      V_ = 0; W_ = 0
      for q in range(len(S_)):
        V_ += mutual_info_classif(x.loc[:,S_[q]].values.reshape(-1,1),y.values,n_neighbors=5,random_state=321)
        for p in range(len(S_)):
          W_ += mutual_info_regression(x.loc[:,S_[q]].values.reshape(-1,1),x.loc[:,S_[p]].values,n_neighbors=5,random_state=321)
      V = V_/len(S_); W = W_/(len(S_)**2)
      MIQ.append(V/W); MID.append(V-W)
    if method =="MIQ":
      best_idx = np.argmax(MIQ)
      S.append(F[best_idx]);F.drop(F[best_idx]);mRMR.append(np.max(MIQ)); t += 1
    else:
      best_idx = np.argmax(MID)
      S.append(F[best_idx]);F.drop(F[best_idx]);mRMR.append(np.max(MID)); t += 1
    if t%2 ==0: print(t)

  return S, mRMR



filtered, results = mRMR_filter(X_tr,y_tr, 16,method="MIQ") #연산량 과다로 시간 많이 소요


#변수선택 및 성능 비교

X_tr_s = X_tr.loc[:,filtered]
X_te_s = X_te.loc[:,filtered]

rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=321)
rf1.fit(X_tr,y_tr)
print(f1_score(y_te,rf1.predict(X_te),average="macro"))

rf2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=321)
rf2.fit(X_tr_s,y_tr)
print(f1_score(y_te,rf2.predict(X_te_s),average="macro"))
