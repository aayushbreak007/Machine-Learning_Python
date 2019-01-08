# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 15:01:02 2018

@author: AAYUSH
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


creditData = pd.read_csv("Z:\\Machine_Learning_Algorithms-Python\\Datasets\\Credit_Data.csv")

#this is used to expand all the columns and rows (hidden)
pd.set_option('expand_frame_repr', False)

# head() will print out the first few (top 5 ) rows in the dataset
print(creditData.head(),"\n")

# describe() will give the statistical data info regarding the data coulmns (mean,variance,count etc)
print(creditData.describe(),"\n")

#this will return a correlation matrix --describing relations b/w 2 features
# the correlation b/w 2 variables (x,y)= cov(x,y)/sd of x * sd of y
# it tells correlation in matrix form "positive/negative/no relation
print(creditData.corr(),"\n")

# making our own selected features dataset
# selecting the features from the dataset
features=creditData[["income","age","loan"]]
# selecting the target column which we need to "PREDICT" from the dataset i.e defaulter(default)
target=creditData.default



#*********************************************************************************************
# now we use some portion of the dataset fro "training" purposes and the rest for the "test"
# we are going to use 30% of the data in the datset is for testing and 70%is for training

feature_train,feature_test,target_train,target_test=train_test_split(features,target,test_size=0.3)


#***********************************************************************************************



# training the feature and target 70% with the model
# maximum liklihood estimator is going to estimate the optimal b0 ,b1, b2 parameters(income,age,loan)
model=LogisticRegression(solver='lbfgs')
model.fit=model.fit(feature_train,target_train)

#predictions with the feature_tests
predictions=model.fit.predict(feature_test)



#testing the accuracy with the help of "Confusion matrix" and accuracy score
print("confusion matrix:\n",confusion_matrix(target_test,predictions))
print("accuracy:",accuracy_score(target_test,predictions))






