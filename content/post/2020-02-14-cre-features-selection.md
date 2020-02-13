---
title: CRE features selection
author: Hermit
date: '2020-02-14'
slug: cre-features-selection
categories:
  - machine-learning
  - Python
tags:
  - classification
---
This time I will use the scikit-learn module to bulid the classifiers,and I will use the randomforest's importance to choose the explanatory variables.   
![](/post/2020-02-14-cre-features-selection_files/1.JPG)
# Part1. Import the data and R'S randomforest importance


```python
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/User/OneDrive - student.nsysu.edu.tw/Educations/NSYSU/fu_chung/bacterial/123.csv')
```


```python
impor = pd.read_csv('C:/Users/User/OneDrive - student.nsysu.edu.tw/Educations/NSYSU/fu_chung/bacterial - PCA/A.csv')
impo = np.array(impor['names'])
impo
```




    array(['V994', 'V1428', 'V1426', ..., 'V1469', 'V1470', 'V1471'],
          dtype=object)



# Part2. Classifiers Building
Contain methods:  
svm,randomforest,navie bayes,knn,lda,qda,adaboost,logistic regression.  

Those function will return the:
"Methods Name",  
"All True amount",  
"Whole Accuracy",  
"True CRE amount",  
"True non-CRE amount",  
"CRE Accuracy",  
"non-CRE Accuracy".  


```python
from sklearn import ensemble
from sklearn import metrics
from sklearn import svm 
#SVM
def svmloocv(ldf):
    ldf = ldf.reset_index(drop=True)
    cv = []
    for i in range(len(ldf)):
        dtrain = ldf.drop([i])
        dtest = ldf.iloc[i:i+1,:]
        train_X = dtrain.iloc[:,0:ldf.shape[1]-1]
        test_X = dtest.iloc[:,0:ldf.shape[1]-1]
        train_y = dtrain["CRE"]
        test_y = dtest["CRE"]
        clf = svm.SVC(kernel = 'linear') #SVM模組，svc,線性核函式 
        clf_fit = clf.fit(train_X, train_y)
        test_y_predicted = clf.predict(test_X)
        accuracy_rf = metrics.accuracy_score(test_y, test_y_predicted)
        cv += [accuracy_rf]
    loocv = np.mean(cv)
    return "SupportVectorMachine",sum(cv),loocv,sum(cv[0:46]),sum(cv[46:95]),sum(cv[0:46])/46,sum(cv[46:95])/49
#RF
def rfloocv(ldf):
    ldf = ldf.reset_index(drop=True)
    cv = []
    for i in range(len(ldf)):
        dtrain = ldf.drop([i])
        dtest = ldf.iloc[i:i+1,:]
        train_X = dtrain.iloc[:,0:ldf.shape[1]-1]
        test_X = dtest.iloc[:,0:ldf.shape[1]-1]
        train_y = dtrain["CRE"]
        test_y = dtest["CRE"]
        clf = ensemble.RandomForestClassifier(n_estimators = 10)
        clf_fit = clf.fit(train_X, train_y)
        test_y_predicted = clf.predict(test_X)
        accuracy_rf = metrics.accuracy_score(test_y, test_y_predicted)
        cv += [accuracy_rf]
    loocv = np.mean(cv)
    return "RandomForest",sum(cv),loocv,sum(cv[0:46]),sum(cv[46:95]),sum(cv[0:46])/46,sum(cv[46:95])/49
#NB
def nbloocv(ldf):
    ldf = ldf.reset_index(drop=True)
    cv = []
    for i in range(len(ldf)):
        dtrain = ldf.drop([i])
        dtest = ldf.iloc[i:i+1,:]
        train_X = dtrain.iloc[:,0:ldf.shape[1]-1]
        test_X = dtest.iloc[:,0:ldf.shape[1]-1]
        train_y = dtrain["CRE"]
        test_y = dtest["CRE"]
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf_fit = clf.fit(train_X, train_y)
        test_y_predicted = clf.predict(test_X)
        accuracy_rf = metrics.accuracy_score(test_y, test_y_predicted)
        cv += [accuracy_rf]
    loocv = np.mean(cv)
    return "NaiveBayes",sum(cv),loocv,sum(cv[0:46]),sum(cv[46:95]),sum(cv[0:46])/46,sum(cv[46:95])/49
#KNN
def knnloocv(ldf):
    ldf = ldf.reset_index(drop=True)
    cv = []
    for i in range(len(ldf)):
        dtrain = ldf.drop([i])
        dtest = ldf.iloc[i:i+1,:]
        train_X = dtrain.iloc[:,0:ldf.shape[1]-1]
        test_X = dtest.iloc[:,0:ldf.shape[1]-1]
        train_y = dtrain["CRE"]
        test_y = dtest["CRE"]
        from sklearn.neighbors import KNeighborsClassifier 
        clf = KNeighborsClassifier(n_neighbors=3)
        clf_fit = clf.fit(train_X, train_y)
        test_y_predicted = clf.predict(test_X)
        accuracy_rf = metrics.accuracy_score(test_y, test_y_predicted)
        cv += [accuracy_rf]
    loocv = np.mean(cv)
    return "KNN",sum(cv),loocv,sum(cv[0:46]),sum(cv[46:95]),sum(cv[0:46])/46,sum(cv[46:95])/49
#LDA
def ldaloocv(ldf):
    ldf = ldf.reset_index(drop=True)
    cv = []
    for i in range(len(ldf)):
        dtrain = ldf.drop([i])
        dtest = ldf.iloc[i:i+1,:]
        train_X = dtrain.iloc[:,0:ldf.shape[1]-1]
        test_X = dtest.iloc[:,0:ldf.shape[1]-1]
        train_y = dtrain["CRE"]
        test_y = dtest["CRE"]
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None, priors=None)
        clf_fit = clf.fit(train_X, train_y)
        test_y_predicted = clf.predict(test_X)
        accuracy_rf = metrics.accuracy_score(test_y, test_y_predicted)
        cv += [accuracy_rf]
    loocv = np.mean(cv)
    return "LDA",sum(cv),loocv,sum(cv[0:46]),sum(cv[46:95]),sum(cv[0:46])/46,sum(cv[46:95])/49
#QDA
def qdaloocv(ldf):
    ldf = ldf.reset_index(drop=True)
    cv = []
    for i in range(len(ldf)):
        dtrain = ldf.drop([i])
        dtest = ldf.iloc[i:i+1,:]
        train_X = dtrain.iloc[:,0:ldf.shape[1]-1]
        test_X = dtest.iloc[:,0:ldf.shape[1]-1]
        train_y = dtrain["CRE"]
        test_y = dtest["CRE"]
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        clf = QuadraticDiscriminantAnalysis()
        clf_fit = clf.fit(train_X, train_y)
        test_y_predicted = clf.predict(test_X)
        accuracy_rf = metrics.accuracy_score(test_y, test_y_predicted)
        cv += [accuracy_rf]
    loocv = np.mean(cv)
    return "QDA",sum(cv),loocv,sum(cv[0:46]),sum(cv[46:95]),sum(cv[0:46])/46,sum(cv[46:95])/49
#ADABOOST
def adaloocv(ldf):
    ldf = ldf.reset_index(drop=True)
    cv = []
    for i in range(len(ldf)):
        dtrain = ldf.drop([i])
        dtest = ldf.iloc[i:i+1,:]
        train_X = dtrain.iloc[:,0:ldf.shape[1]-1]
        test_X = dtest.iloc[:,0:ldf.shape[1]-1]
        train_y = dtrain["CRE"]
        test_y = dtest["CRE"]
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators=100)
        clf_fit = clf.fit(train_X, train_y)
        test_y_predicted = clf.predict(test_X)
        accuracy_rf = metrics.accuracy_score(test_y, test_y_predicted)
        cv += [accuracy_rf]
    loocv = np.mean(cv)
    return "adaboost",sum(cv),loocv,sum(cv[0:46]),sum(cv[46:95]),sum(cv[0:46])/46,sum(cv[46:95])/49
#logistic 
def glmloocv(ldf):
    ldf = ldf.reset_index(drop=True)
    cv = []
    for i in range(len(ldf)):
        dtrain = ldf.drop([i])
        dtest = ldf.iloc[i:i+1,:]
        train_X = dtrain.iloc[:,0:ldf.shape[1]-1]
        test_X = dtest.iloc[:,0:ldf.shape[1]-1]
        train_y = dtrain["CRE"]
        test_y = dtest["CRE"]
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1000, random_state=0)
        clf_fit = clf.fit(train_X, train_y)
        test_y_predicted = clf.predict(test_X)
        accuracy_rf = metrics.accuracy_score(test_y, test_y_predicted)
        cv += [accuracy_rf]
    loocv = np.mean(cv)
    return "LogisticRegression",sum(cv),loocv,sum(cv[0:46]),sum(cv[46:95]),sum(cv[0:46])/46,sum(cv[46:95])/49
```

## Function Testing


```python
ldf = df.loc[:,impo[0:3]]
ldf['CRE'] = df['CRE']
knnloocv(ldf)
```




    ('KNN',
     50.0,
     0.5263157894736842,
     45.0,
     5.0,
     0.9782608695652174,
     0.10204081632653061)



# Part 3. Processing


```python
import time
import sys

lsvm = []
for i in range (20):  
    ldf = df.loc[:,impo[0:30+i]]
    ldf['CRE'] = df['CRE']
    lsvm += [svmloocv(ldf)]
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*i, (100/(20-1))*i))
    sys.stdout.flush()
    time.sleep(0.00000000000001)
lada = []
for i in range (50):  
    ldf = df.loc[:,impo[0:1+i]]
    ldf['CRE'] = df['CRE']
    lada += [adaloocv(ldf)]
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*i, (100/(50-1))*i))
    sys.stdout.flush()
    time.sleep(0.00000000000001)
llda = []
for i in range (50):  
    ldf = df.loc[:,impo[0:1+i]]
    ldf['CRE'] = df['CRE']
    llda += [ldaloocv(ldf)]
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*i, (100/(50-1))*i))
    sys.stdout.flush()
    time.sleep(0.00000000000001)
lqda = []
for i in range (50):  
    ldf = df.loc[:,impo[0:1+i]]
    ldf['CRE'] = df['CRE']
    lqda += [qdaloocv(ldf)]
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*i, (100/(50-1))*i))
    sys.stdout.flush()
    time.sleep(0.00000000000001)
lrf = []
for i in range (50):  
    ldf = df.loc[:,impo[0:1+i]]
    ldf['CRE'] = df['CRE']
    lrf += [rfloocv(ldf)]
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*i, (100/(50-1))*i))
    sys.stdout.flush()
    time.sleep(0.00000000000001)
lnb = []
for i in range (50):  
    ldf = df.loc[:,impo[0:1+i]]
    ldf['CRE'] = df['CRE']
    lnb += [nbloocv(ldf)]
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*i, (100/(50-1))*i))
    sys.stdout.flush()
    time.sleep(0.00000000000001)
lglm = []
for i in range (50):  
    ldf = df.loc[:,impo[0:1+i]]
    ldf['CRE'] = df['CRE']
    lglm += [glmloocv(ldf)]
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*i, (100/(50-1))*i))
    sys.stdout.flush()
    time.sleep(0.00000000000001)
lknn = []
for i in range (50):  
    ldf = df.loc[:,impo[0:1+i]]
    ldf['CRE'] = df['CRE']
    lknn += [knnloocv(ldf)]
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*i, (100/(50-1))*i))
    sys.stdout.flush()
    time.sleep(0.00000000000001)
```

    [================================================= ] 100%
    [================================================= ] 100%
    [================================================= ] 100%
    [================================================= ] 100%
    [================================================= ] 100%
    [================================================= ] 100%
    [================================================= ] 100%
    [================================================= ] 100%


```python
data = pd.concat([pd.DataFrame(lsvm),pd.DataFrame(lada),pd.DataFrame(lrf),pd.DataFrame(lnb),pd.DataFrame(llda),pd.DataFrame(lqda),pd.DataFrame(lknn),pd.DataFrame(lglm)], axis=1)
#Write the csv
data.to_csv("locv.csv",index=False,sep=',')
```




```python
pd.read_csv('C:/Users/User/OneDrive - student.nsysu.edu.tw/Educations/NSYSU/fu_chung/CRE features selection/locv1.csv')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_cre</th>
      <th>num_non</th>
      <th>num_rf_impo</th>
      <th>method</th>
      <th>right_pred_num</th>
      <th>acc</th>
      <th>right_pred_cre_num</th>
      <th>right_pred_ncre_num</th>
      <th>right_pred_cre_acc</th>
      <th>right_pred_ncre_acc</th>
      <th>...</th>
      <th>right_pred_ncre_num.6</th>
      <th>right_pred_cre_acc.6</th>
      <th>right_pred_ncre_acc.6</th>
      <th>method.7</th>
      <th>right_pred_num.7</th>
      <th>acc.7</th>
      <th>right_pred_cre_num.7</th>
      <th>right_pred_ncre_num.7</th>
      <th>right_pred_cre_acc.7</th>
      <th>right_pred_ncre_acc.7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46</td>
      <td>49</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5</td>
      <td>0.978261</td>
      <td>0.102041</td>
      <td>LogisticRegression</td>
      <td>48</td>
      <td>0.505263</td>
      <td>43</td>
      <td>5</td>
      <td>0.934783</td>
      <td>0.102041</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46</td>
      <td>49</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5</td>
      <td>0.978261</td>
      <td>0.102041</td>
      <td>LogisticRegression</td>
      <td>48</td>
      <td>0.505263</td>
      <td>43</td>
      <td>5</td>
      <td>0.934783</td>
      <td>0.102041</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46</td>
      <td>49</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5</td>
      <td>0.978261</td>
      <td>0.102041</td>
      <td>LogisticRegression</td>
      <td>48</td>
      <td>0.505263</td>
      <td>43</td>
      <td>5</td>
      <td>0.934783</td>
      <td>0.102041</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46</td>
      <td>49</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>10</td>
      <td>0.978261</td>
      <td>0.204082</td>
      <td>LogisticRegression</td>
      <td>52</td>
      <td>0.547368</td>
      <td>42</td>
      <td>10</td>
      <td>0.913043</td>
      <td>0.204082</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46</td>
      <td>49</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>32</td>
      <td>0.804348</td>
      <td>0.653061</td>
      <td>LogisticRegression</td>
      <td>52</td>
      <td>0.547368</td>
      <td>34</td>
      <td>18</td>
      <td>0.739130</td>
      <td>0.367347</td>
    </tr>
    <tr>
      <th>5</th>
      <td>46</td>
      <td>49</td>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>32</td>
      <td>0.804348</td>
      <td>0.653061</td>
      <td>LogisticRegression</td>
      <td>52</td>
      <td>0.547368</td>
      <td>34</td>
      <td>18</td>
      <td>0.739130</td>
      <td>0.367347</td>
    </tr>
    <tr>
      <th>6</th>
      <td>46</td>
      <td>49</td>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>34</td>
      <td>0.804348</td>
      <td>0.693878</td>
      <td>LogisticRegression</td>
      <td>54</td>
      <td>0.568421</td>
      <td>33</td>
      <td>21</td>
      <td>0.717391</td>
      <td>0.428571</td>
    </tr>
    <tr>
      <th>7</th>
      <td>46</td>
      <td>49</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>38</td>
      <td>0.891304</td>
      <td>0.775510</td>
      <td>LogisticRegression</td>
      <td>56</td>
      <td>0.589474</td>
      <td>39</td>
      <td>17</td>
      <td>0.847826</td>
      <td>0.346939</td>
    </tr>
    <tr>
      <th>8</th>
      <td>46</td>
      <td>49</td>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>37</td>
      <td>0.891304</td>
      <td>0.755102</td>
      <td>LogisticRegression</td>
      <td>56</td>
      <td>0.589474</td>
      <td>39</td>
      <td>17</td>
      <td>0.847826</td>
      <td>0.346939</td>
    </tr>
    <tr>
      <th>9</th>
      <td>46</td>
      <td>49</td>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>37</td>
      <td>0.891304</td>
      <td>0.755102</td>
      <td>LogisticRegression</td>
      <td>56</td>
      <td>0.589474</td>
      <td>39</td>
      <td>17</td>
      <td>0.847826</td>
      <td>0.346939</td>
    </tr>
    <tr>
      <th>10</th>
      <td>46</td>
      <td>49</td>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>37</td>
      <td>0.891304</td>
      <td>0.755102</td>
      <td>LogisticRegression</td>
      <td>59</td>
      <td>0.621053</td>
      <td>39</td>
      <td>20</td>
      <td>0.847826</td>
      <td>0.408163</td>
    </tr>
    <tr>
      <th>11</th>
      <td>46</td>
      <td>49</td>
      <td>12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>41</td>
      <td>0.760870</td>
      <td>0.836735</td>
      <td>LogisticRegression</td>
      <td>60</td>
      <td>0.631579</td>
      <td>44</td>
      <td>16</td>
      <td>0.956522</td>
      <td>0.326531</td>
    </tr>
    <tr>
      <th>12</th>
      <td>46</td>
      <td>49</td>
      <td>13</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>44</td>
      <td>0.847826</td>
      <td>0.897959</td>
      <td>LogisticRegression</td>
      <td>60</td>
      <td>0.631579</td>
      <td>45</td>
      <td>15</td>
      <td>0.978261</td>
      <td>0.306122</td>
    </tr>
    <tr>
      <th>13</th>
      <td>46</td>
      <td>49</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>44</td>
      <td>0.847826</td>
      <td>0.897959</td>
      <td>LogisticRegression</td>
      <td>60</td>
      <td>0.631579</td>
      <td>45</td>
      <td>15</td>
      <td>0.978261</td>
      <td>0.306122</td>
    </tr>
    <tr>
      <th>14</th>
      <td>46</td>
      <td>49</td>
      <td>15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>44</td>
      <td>0.826087</td>
      <td>0.897959</td>
      <td>LogisticRegression</td>
      <td>61</td>
      <td>0.642105</td>
      <td>45</td>
      <td>16</td>
      <td>0.978261</td>
      <td>0.326531</td>
    </tr>
    <tr>
      <th>15</th>
      <td>46</td>
      <td>49</td>
      <td>16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>44</td>
      <td>0.826087</td>
      <td>0.897959</td>
      <td>LogisticRegression</td>
      <td>59</td>
      <td>0.621053</td>
      <td>44</td>
      <td>15</td>
      <td>0.956522</td>
      <td>0.306122</td>
    </tr>
    <tr>
      <th>16</th>
      <td>46</td>
      <td>49</td>
      <td>17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>44</td>
      <td>0.826087</td>
      <td>0.897959</td>
      <td>LogisticRegression</td>
      <td>65</td>
      <td>0.684211</td>
      <td>46</td>
      <td>19</td>
      <td>1.000000</td>
      <td>0.387755</td>
    </tr>
    <tr>
      <th>17</th>
      <td>46</td>
      <td>49</td>
      <td>18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>45</td>
      <td>0.826087</td>
      <td>0.918367</td>
      <td>LogisticRegression</td>
      <td>61</td>
      <td>0.642105</td>
      <td>46</td>
      <td>15</td>
      <td>1.000000</td>
      <td>0.306122</td>
    </tr>
    <tr>
      <th>18</th>
      <td>46</td>
      <td>49</td>
      <td>19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>45</td>
      <td>0.826087</td>
      <td>0.918367</td>
      <td>LogisticRegression</td>
      <td>60</td>
      <td>0.631579</td>
      <td>45</td>
      <td>15</td>
      <td>0.978261</td>
      <td>0.306122</td>
    </tr>
    <tr>
      <th>19</th>
      <td>46</td>
      <td>49</td>
      <td>20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>46</td>
      <td>0.826087</td>
      <td>0.938776</td>
      <td>LogisticRegression</td>
      <td>61</td>
      <td>0.642105</td>
      <td>44</td>
      <td>17</td>
      <td>0.956522</td>
      <td>0.346939</td>
    </tr>
    <tr>
      <th>20</th>
      <td>46</td>
      <td>49</td>
      <td>21</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>45</td>
      <td>0.804348</td>
      <td>0.918367</td>
      <td>LogisticRegression</td>
      <td>61</td>
      <td>0.642105</td>
      <td>44</td>
      <td>17</td>
      <td>0.956522</td>
      <td>0.346939</td>
    </tr>
    <tr>
      <th>21</th>
      <td>46</td>
      <td>49</td>
      <td>22</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>45</td>
      <td>0.804348</td>
      <td>0.918367</td>
      <td>LogisticRegression</td>
      <td>65</td>
      <td>0.684211</td>
      <td>42</td>
      <td>23</td>
      <td>0.913043</td>
      <td>0.469388</td>
    </tr>
    <tr>
      <th>22</th>
      <td>46</td>
      <td>49</td>
      <td>23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>44</td>
      <td>0.804348</td>
      <td>0.897959</td>
      <td>LogisticRegression</td>
      <td>67</td>
      <td>0.705263</td>
      <td>41</td>
      <td>26</td>
      <td>0.891304</td>
      <td>0.530612</td>
    </tr>
    <tr>
      <th>23</th>
      <td>46</td>
      <td>49</td>
      <td>24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>44</td>
      <td>0.804348</td>
      <td>0.897959</td>
      <td>LogisticRegression</td>
      <td>67</td>
      <td>0.705263</td>
      <td>42</td>
      <td>25</td>
      <td>0.913043</td>
      <td>0.510204</td>
    </tr>
    <tr>
      <th>24</th>
      <td>46</td>
      <td>49</td>
      <td>25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>44</td>
      <td>0.804348</td>
      <td>0.897959</td>
      <td>LogisticRegression</td>
      <td>64</td>
      <td>0.673684</td>
      <td>40</td>
      <td>24</td>
      <td>0.869565</td>
      <td>0.489796</td>
    </tr>
    <tr>
      <th>25</th>
      <td>46</td>
      <td>49</td>
      <td>26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>44</td>
      <td>0.804348</td>
      <td>0.897959</td>
      <td>LogisticRegression</td>
      <td>66</td>
      <td>0.694737</td>
      <td>41</td>
      <td>25</td>
      <td>0.891304</td>
      <td>0.510204</td>
    </tr>
    <tr>
      <th>26</th>
      <td>46</td>
      <td>49</td>
      <td>27</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>45</td>
      <td>0.804348</td>
      <td>0.918367</td>
      <td>LogisticRegression</td>
      <td>68</td>
      <td>0.715789</td>
      <td>40</td>
      <td>28</td>
      <td>0.869565</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>27</th>
      <td>46</td>
      <td>49</td>
      <td>28</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>45</td>
      <td>0.804348</td>
      <td>0.918367</td>
      <td>LogisticRegression</td>
      <td>66</td>
      <td>0.694737</td>
      <td>38</td>
      <td>28</td>
      <td>0.826087</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>28</th>
      <td>46</td>
      <td>49</td>
      <td>29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>45</td>
      <td>0.804348</td>
      <td>0.918367</td>
      <td>LogisticRegression</td>
      <td>68</td>
      <td>0.715789</td>
      <td>40</td>
      <td>28</td>
      <td>0.869565</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>29</th>
      <td>46</td>
      <td>49</td>
      <td>30</td>
      <td>SupportVectorMachine</td>
      <td>87.0</td>
      <td>0.915789</td>
      <td>42.0</td>
      <td>45.0</td>
      <td>0.913043</td>
      <td>0.918367</td>
      <td>...</td>
      <td>46</td>
      <td>0.826087</td>
      <td>0.938776</td>
      <td>LogisticRegression</td>
      <td>69</td>
      <td>0.726316</td>
      <td>40</td>
      <td>29</td>
      <td>0.869565</td>
      <td>0.591837</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>219</th>
      <td>46</td>
      <td>49</td>
      <td>220</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>89</td>
      <td>0.936842</td>
      <td>43</td>
      <td>46</td>
      <td>0.934783</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>220</th>
      <td>46</td>
      <td>49</td>
      <td>221</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>89</td>
      <td>0.936842</td>
      <td>43</td>
      <td>46</td>
      <td>0.934783</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>221</th>
      <td>46</td>
      <td>49</td>
      <td>222</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>89</td>
      <td>0.936842</td>
      <td>43</td>
      <td>46</td>
      <td>0.934783</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>222</th>
      <td>46</td>
      <td>49</td>
      <td>223</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>88</td>
      <td>0.926316</td>
      <td>42</td>
      <td>46</td>
      <td>0.913043</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>223</th>
      <td>46</td>
      <td>49</td>
      <td>224</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>89</td>
      <td>0.936842</td>
      <td>43</td>
      <td>46</td>
      <td>0.934783</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>224</th>
      <td>46</td>
      <td>49</td>
      <td>225</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>89</td>
      <td>0.936842</td>
      <td>43</td>
      <td>46</td>
      <td>0.934783</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>225</th>
      <td>46</td>
      <td>49</td>
      <td>226</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>88</td>
      <td>0.926316</td>
      <td>42</td>
      <td>46</td>
      <td>0.913043</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>226</th>
      <td>46</td>
      <td>49</td>
      <td>227</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>89</td>
      <td>0.936842</td>
      <td>43</td>
      <td>46</td>
      <td>0.934783</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>227</th>
      <td>46</td>
      <td>49</td>
      <td>228</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>89</td>
      <td>0.936842</td>
      <td>43</td>
      <td>46</td>
      <td>0.934783</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>228</th>
      <td>46</td>
      <td>49</td>
      <td>229</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>89</td>
      <td>0.936842</td>
      <td>43</td>
      <td>46</td>
      <td>0.934783</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>229</th>
      <td>46</td>
      <td>49</td>
      <td>230</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>88</td>
      <td>0.926316</td>
      <td>42</td>
      <td>46</td>
      <td>0.913043</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>230</th>
      <td>46</td>
      <td>49</td>
      <td>231</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>87</td>
      <td>0.915789</td>
      <td>42</td>
      <td>45</td>
      <td>0.913043</td>
      <td>0.918367</td>
    </tr>
    <tr>
      <th>231</th>
      <td>46</td>
      <td>49</td>
      <td>232</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>87</td>
      <td>0.915789</td>
      <td>42</td>
      <td>45</td>
      <td>0.913043</td>
      <td>0.918367</td>
    </tr>
    <tr>
      <th>232</th>
      <td>46</td>
      <td>49</td>
      <td>233</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>87</td>
      <td>0.915789</td>
      <td>42</td>
      <td>45</td>
      <td>0.913043</td>
      <td>0.918367</td>
    </tr>
    <tr>
      <th>233</th>
      <td>46</td>
      <td>49</td>
      <td>234</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>88</td>
      <td>0.926316</td>
      <td>42</td>
      <td>46</td>
      <td>0.913043</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>234</th>
      <td>46</td>
      <td>49</td>
      <td>235</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>88</td>
      <td>0.926316</td>
      <td>42</td>
      <td>46</td>
      <td>0.913043</td>
      <td>0.938776</td>
    </tr>
    <tr>
      <th>235</th>
      <td>46</td>
      <td>49</td>
      <td>236</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.847826</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>87</td>
      <td>0.915789</td>
      <td>42</td>
      <td>45</td>
      <td>0.913043</td>
      <td>0.918367</td>
    </tr>
    <tr>
      <th>236</th>
      <td>46</td>
      <td>49</td>
      <td>237</td>
      <td>SupportVectorMachine</td>
      <td>90.0</td>
      <td>0.947368</td>
      <td>42.0</td>
      <td>48.0</td>
      <td>0.913043</td>
      <td>0.979592</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>237</th>
      <td>46</td>
      <td>49</td>
      <td>238</td>
      <td>SupportVectorMachine</td>
      <td>90.0</td>
      <td>0.947368</td>
      <td>42.0</td>
      <td>48.0</td>
      <td>0.913043</td>
      <td>0.979592</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>238</th>
      <td>46</td>
      <td>49</td>
      <td>239</td>
      <td>SupportVectorMachine</td>
      <td>90.0</td>
      <td>0.947368</td>
      <td>42.0</td>
      <td>48.0</td>
      <td>0.913043</td>
      <td>0.979592</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>239</th>
      <td>46</td>
      <td>49</td>
      <td>240</td>
      <td>SupportVectorMachine</td>
      <td>90.0</td>
      <td>0.947368</td>
      <td>42.0</td>
      <td>48.0</td>
      <td>0.913043</td>
      <td>0.979592</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>240</th>
      <td>46</td>
      <td>49</td>
      <td>241</td>
      <td>SupportVectorMachine</td>
      <td>90.0</td>
      <td>0.947368</td>
      <td>42.0</td>
      <td>48.0</td>
      <td>0.913043</td>
      <td>0.979592</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>241</th>
      <td>46</td>
      <td>49</td>
      <td>242</td>
      <td>SupportVectorMachine</td>
      <td>90.0</td>
      <td>0.947368</td>
      <td>42.0</td>
      <td>48.0</td>
      <td>0.913043</td>
      <td>0.979592</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>242</th>
      <td>46</td>
      <td>49</td>
      <td>243</td>
      <td>SupportVectorMachine</td>
      <td>90.0</td>
      <td>0.947368</td>
      <td>42.0</td>
      <td>48.0</td>
      <td>0.913043</td>
      <td>0.979592</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>243</th>
      <td>46</td>
      <td>49</td>
      <td>244</td>
      <td>SupportVectorMachine</td>
      <td>90.0</td>
      <td>0.947368</td>
      <td>42.0</td>
      <td>48.0</td>
      <td>0.913043</td>
      <td>0.979592</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>244</th>
      <td>46</td>
      <td>49</td>
      <td>245</td>
      <td>SupportVectorMachine</td>
      <td>89.0</td>
      <td>0.936842</td>
      <td>42.0</td>
      <td>47.0</td>
      <td>0.913043</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>245</th>
      <td>46</td>
      <td>49</td>
      <td>246</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>246</th>
      <td>46</td>
      <td>49</td>
      <td>247</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>247</th>
      <td>46</td>
      <td>49</td>
      <td>248</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
    <tr>
      <th>248</th>
      <td>46</td>
      <td>49</td>
      <td>249</td>
      <td>SupportVectorMachine</td>
      <td>88.0</td>
      <td>0.926316</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>0.891304</td>
      <td>0.959184</td>
      <td>...</td>
      <td>48</td>
      <td>0.869565</td>
      <td>0.979592</td>
      <td>LogisticRegression</td>
      <td>90</td>
      <td>0.947368</td>
      <td>43</td>
      <td>47</td>
      <td>0.934783</td>
      <td>0.959184</td>
    </tr>
  </tbody>
</table>
<p>249 rows × 59 columns</p>
</div>



All of the above results, the best accuracy is adaboost in the interval between 67~166. Its accuracy in 0.989473684. Only miss one prediction.