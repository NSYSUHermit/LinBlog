---
title: MLB playoff prediction with SVM & Randomforest
author: Hermit
date: '2019-08-19'
slug: mlb-playoff-prediction
categories:
  - Python
  - machine-learning
tags:
  - classification
---
![mlb](/post/2019-05-02-web-crawler-on-simple-chinese-web_files/2018-07-17_19-00-33.png)
In the same theme,MLB.   
I will show how to use Python to train a randomforest and SVM classifier.  
Our target is predicting whether the team will enter the next year playoff or not.
Since the playoff qualification depends on the same year's season game win rate.
If we predict the same year playoff is pointless.
So I make the variable "palyoff" to "next year playoff".
I will use the module "sklearn" to finish my work.

# Import module
```python
import numpy as np
import pandas as pd
from sklearn import cross_validation, ensemble, preprocessing, metrics 
```
    

# Read predict data
```python
data = pd.read_csv('C:/Users/User/OneDrive - student.nsysu.edu.tw/Educations/2019暑期實習/工研院/資料科學_mlb/mlb_predict.csv')
```

# Build Train and Test data set
I choose the last article regression model variable. 
That will become more sensible since the playoff relate with win rate.  
And I cut the data on 7:3 to train set and test set.
```python
mlb_X = pd.DataFrame([data["Bat"],data["AB"],data["HR_x"],data["SO_x"],data["OPS+"],data["GDP"],
                      data["IBB_x"],data["#P"],data["PAge"],data["tSho"],data["SV"],data["IP"],
                      data["HBP_y"],data["ERA+"],data["WP"]]).T
mlb_y = data["Playoff_next"]
train_X, test_X, train_y, test_y = cross_validation.train_test_split(mlb_X, mlb_y, test_size = 0.3)
```
# Random Forest
## Build the random forest model
I chose to use 100 trees to build the model this time.
```python
forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(train_X, train_y)
```

## Predict on random forest model
```python
test_y_predicted = forest.predict(test_X)
accuracy_rf = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy_rf)

```

    0.7350427350427351

It shows that the accuracy is 0.7350427350427351.  

# SVM

## Build SVC model
I didn't set any parameter this time. All is default.
```python
from sklearn import cross_validation, svm, preprocessing, metrics
svc = svm.SVC()
svc_fit = svc.fit(train_X, train_y)
```

## Predict on random forest model
```python
# Predict
test_y_predicted = svc.predict(test_X)
accuracy_svm = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy_svm)
```

    0.7521367521367521
    
It shows that the accuracy is 0.7521367521367521.  