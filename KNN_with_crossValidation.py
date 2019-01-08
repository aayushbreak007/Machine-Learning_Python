# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 18:05:28 2018

@author: AAYUSH
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

creditData=pd.read_csv("Z:\\Machine_Learning_Algorithms-Python\\Datasets\\Credit_Data.csv")

features = creditData[["income","age","loan"]] #--input
target = creditData.default #--output

print(features.head())

# before applying KNN we need to pre-process dataset to normalize the data values
# in the range 0 to 1 by applying "min-max normalization" or z-transformation
features=preprocessing.MinMaxScaler().fit_transform(features)

# split the data-set for training and testing
feature_train, feature_test, target_train, target_test = train_test_split(features,target, test_size=0.3)

# now we need to find the optimal 'K' value for KNN classifier
# so we use cross_validation----K-folds cross validation 
cross_validation_scores=[]
for k in range(1,100):
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,features,target,cv=10,scoring='accuracy')
    #storing the score mean at each iteration
    cross_validation_scores.append(scores.mean())
    
    """ the above statement stores the optimal K value that we should use for best accuracy
     in the model"""
    
# now fitting the model with optimal k value--which is the max value(accuracy score) in the array
print("score:",np.argmax(cross_validation_scores))     
model=KNeighborsClassifier(n_neighbors=np.argmax(cross_validation_scores))
fittedModel=model.fit(feature_train,target_train)
predictions=fittedModel.predict(feature_test)

print("confusion matrix:\n",confusion_matrix(target_test,predictions))
print("accuracy:",accuracy_score(target_test,predictions))




