# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:00:17 2018

@author: AAYUSH
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("Z:\\Machine_Learning_Algorithms-Python\\Datasets\\Credit_Data.csv")


# KNN out-performs naive bayes and logistic regression withtout any underlying model


features = data[["income","age","loan"]]
target = data.default

feature_train, feature_test, target_train, target_test = train_test_split(features,target, test_size=0.3)

model = GaussianNB() #gaussian naive bayes ---gaussian means that features are normal distributed
fittedModel = model.fit(feature_train, target_train)
predictions = fittedModel.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))