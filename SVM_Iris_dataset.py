# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:26:19 2018

@author: AAYUSH
"""

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets


#
#	Important parameters for SVC: gamma and C
#		gamma -> defines how far the influence of a single training example reaches
#					Low value: influence reaches far      High value: influence reaches close
#
#		C -> trades off hyperplane surface simplicity + training examples missclassifications
#					Low value: simple/smooth hyperplane surface 
#					High value: all training examples classified correctly but complex surface 

dataset = datasets.load_iris()

#print(dataset)

features = dataset.data
targetVariables = dataset.target

featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size=0.3)

#model = svm.SVC(gamma=0.0001, C=100)
model = svm.SVC()
fittedModel = model.fit(featureTrain, targetTrain)
predictions = fittedModel.predict(featureTest)


#diagnal items are the correclty classified items and the off-diagonal items are mis-classified
print(confusion_matrix(targetTest, predictions))
print(accuracy_score(targetTest, predictions))
