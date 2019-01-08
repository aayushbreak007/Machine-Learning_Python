# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 23:20:56 2018

@author: AAYUSH
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = pd.read_csv("Z:\\Machine_Learning_Algorithms-Python\\Datasets\\iris_data.csv")

print(data.head())
features = data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
targets = data.Class 

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

'''Gini index approach is faster since we don't have to compute 
log values whcih are expensive'''

#model = DecisionTreeClassifier(criterion='entropy')
model = DecisionTreeClassifier(criterion='gini')
model.fitted = model.fit(feature_train, target_train)
model.predictions = model.fitted.predict(feature_test)

print(confusion_matrix(target_test, model.predictions))
print(accuracy_score(target_test, model.predictions))


#validate preformance with cross validation
scores=cross_val_score(model,features,targets,cv=5)
print("cross-validation:",scores.mean())

