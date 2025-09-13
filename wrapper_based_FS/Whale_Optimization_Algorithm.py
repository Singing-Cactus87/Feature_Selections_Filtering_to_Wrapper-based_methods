import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score, f1_score, precision_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score


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


#Whale Optimization Algorithm 정의


from sklearn.model_selection import cross_validate

def Whale_Opt_Algorithm(x,y,M,K,b,md):
  best_score = []; t=0; model=md
  W_mat = np.random.choice([0,1],size=M*x.shape[1]).reshape(M,-1)
  sc = []

  def binarize_(x):
    bn = 1/(1+np.exp(-x))
    if bn > 0.5: return 1
    else: return 0

  def binarize_3(x): #포스트에서는 해당 함수를 이항화 함수로 활용
    bn = 1/(1+np.exp(-10*(x-0.5)))
    if bn > np.random.rand(1): return 1
    else: return 0

  for i in range(M):
    idx =[bool(w) for w in W_mat[i,:]]
    f_t = x.loc[:,idx].columns
    score = np.mean(cross_validate(model,x.loc[:,f_t],y,cv=3, scoring=['f1'])['test_f1'])
    sc.append(score)
  best_idx = np.argmax(sc); best_score.append(np.max(sc))
  W_best = W_mat[best_idx,:]; sc=[];t += 1
  while t < K:
    a = (np.ones(x.shape[1])+1)-2*t/K
    for j in range(M):
      r = np.random.rand(x.shape[1]); C = 2*r; A = 2*a*r - a; p = np.random.rand(1); l = np.random.uniform(low=-1,high=1,size=1)
      if p < 0.5:
        if np.linalg.norm(A) < 1:
          D = np.abs(C*W_best-W_mat[j,:])
          W_mat[j,:] = W_best -A*D
        else:
          rand = np.random.choice(np.arange(M),1); W_rand = W_mat[rand,:]
          D = np.abs(C*W_rand-W_mat[j,:])
          W_mat[j,:] = W_rand -A*D
      else:
        D1 = np.abs(W_best-W_mat[j,:])
        W_mat[j,:] = D1*np.exp(b*l)*np.cos(2*np.pi*l)+W_best

    for q1 in range(M):
      for q2 in range(x.shape[1]):
        W_mat[q1,q2] = binarize_3(W_mat[q1,q2])

    for q in range(M):
      if np.sum(W_mat[q,:])==0:
        W_mat[q,:] = np.random.choice([0,1],x.shape[1])

    for i in range(M):
      idx =[bool(w) for w in W_mat[i,:]]
      f_t = x.loc[:,idx].columns
      score = np.mean(cross_validate(model,x.loc[:,f_t],y,cv=3, scoring=['f1'])['test_f1'])
      sc.append(score)
    best_idx = np.argmax(sc); best_score.append(np.max(sc))
    W_best = W_mat[best_idx,:]; sc=[];t += 1
    if t %2 ==0: print(t)

  best_bool =[bool(w) for w in W_best]
  W_best_features = x.loc[:,best_bool].columns

  return W_best_features, best_score



#WOA 실행
np.random.seed(345)
SET, results = Whale_Opt_Algorithm(X_tr,y_tr,10,30,1e-2,RandomForestClassifier(n_estimators=100,random_state=321))


X_tr_s = X_tr.loc[:,SET]
X_te_s = X_te.loc[:,SET]


#추출된 변수 개수 확인: 12개
len(SET)


#성능 비교
rf1 = RandomForestClassifier(n_estimators=100,random_state=321)
rf1.fit(X_tr,y_tr)
print(f1_score(y_te,rf1.predict(X_te),average="macro"))
print(accuracy_score(y_te,rf1.predict(X_te)))
print(roc_auc_score(y_te,rf1.predict(X_te)))

rf2 = RandomForestClassifier(n_estimators=100,random_state=321)
rf2.fit(X_tr_s,y_tr)
print(f1_score(y_te,rf2.predict(X_te_s),average="macro"))
print(accuracy_score(y_te,rf2.predict(X_te_s)))
print(roc_auc_score(y_te,rf2.predict(X_te_s)))
