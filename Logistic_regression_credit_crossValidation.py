# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:24:39 2018

@author: AAYUSH
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

creditData = pd.read_csv("Z:\\Machine_Learning_Algorithms-Python\\Datasets\\Credit_Data.csv")

features = creditData[["income","age","loan"]]
target = creditData.default


# instead of splitting the data we will use----K-folds cross validation
# this solves the problem of overfitting and underfitting data model
# cv means partitions
# uses all the dataset for training purposes and testing
model = LogisticRegression(solver='lbfgs')
model.fit(features,target)

#validation score---this is only used to validate how well the model performs
score = cross_val_score(model,features,target, cv=10,scoring='accuracy')

print("accuracy:",score.mean())
print(model.predict_proba([[66952.6888453402,18.5843359269202,8770.09923520439]]))