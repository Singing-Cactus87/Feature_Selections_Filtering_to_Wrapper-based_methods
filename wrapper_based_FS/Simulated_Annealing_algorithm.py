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




#Simulated Annealing 기반 변수선택 함수 정의

from sklearn.model_selection import cross_validate

def SA_wrap(x,y,K,T,md,thres=1e-5):
  Energy = []; t = 0; model = md; T0 =T*1
  S_init = x.iloc[:,[bool(f) for f in np.random.choice([0,1],p=[0.5,0.5],size=x.shape[1])]].columns
  E_init = -np.mean(cross_validate(model,x.loc[:,S_init],y,cv=3, scoring=['f1'])['test_f1'])
  S_current = S_init; E_current = E_init; t += 1; delta_E = E_init-0
  Energy.append(E_current)
  while (t < K and np.abs(delta_E) > thres):
    model = md
    S_cand = x.iloc[:,[bool(f) for f in np.random.choice([0,1],p=[0.5,0.5],size=x.shape[1])]].columns
    E_cand = -np.mean(cross_validate(model,x.loc[:,S_cand],y,cv=3, scoring=['f1'])['test_f1'])
    delta_E = E_cand-E_current
    if delta_E < 0:
      Energy.append(E_cand)
      S_current = S_cand; E_current= E_cand; T = T-(T0/K)
      t += 1
    else:
      u = np.random.uniform(low=0,high=1,size=1)
      if u < np.exp(-delta_E/T):
        Energy.append(E_cand)
        S_current = S_cand; E_current= E_cand; T = T-(T0/K)
        t += 1
      else:
        Energy.append(E_current)
        S_current = S_current; E_current= E_current; T = T-(T0/K)
        t += 1
    if (t%2==0): print(t)

  return S_current, Energy


#SA 기반 변수선택 실행 및 결과 비교

np.random.seed(321)
results, E_ = SA_wrap(X_tr,y_tr,30,0.1,RandomForestClassifier(n_estimators=100,max_depth=10,random_state=321),thres=1e-5)

#SA에서의 step별 Energy 변화 양상
sns.set_style("darkgrid")
plt.plot(np.arange(30),E_,color="darkcyan",label="Energy of SA")
plt.legend()

X_tr_s = X_tr.loc[:,results]
X_te_s = X_te.loc[:,results]

rf1 = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=321)
rf1.fit(X_tr,y_tr)
print(f1_score(y_te,rf1.predict(X_te),average="macro"))


rf2 = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=321)
rf2.fit(X_tr_s,y_tr)
print(f1_score(y_te,rf2.predict(X_te_s),average="macro"))

#차원축소된 데이터셋 차원 확인
X_tr_s.shape

