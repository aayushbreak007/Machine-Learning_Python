# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 20:56:49 2018

@author: AAYUSH
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm


xBlue = np.array([0.3,0.5,1,1.4,1.7,2])
yBlue = np.array([1,4.5,2.3,1.9,8.9,4.1])

xRed = np.array([3.3,3.5,4,4.4,5.7,6])
yRed = np.array([7,1.5,6.3,1.9,2.9,7.1])

X = np.array([[0.3,1],[0.5,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4,6.3],[4.4,1.9],[5.7,2.9],[6,7.1]])
Y = np.array([0,0,0,0,0,0,1,1,1,1,1,1]) # 0: blue class, 1: red class

plt.plot(xBlue, yBlue, 'ro', color='blue')
plt.plot(xRed, yRed, 'ro', color='red')

# point we want to determine lies in which class
plt.plot(2.5,4.5,'ro',color='green') 
plt.plot(6,6,'ro',color='grey')



''' important parameters for svc:
            GAMMA: the hyperplane margin from the support vectors
            C: samples mis-classification:
                if( c is low){
                        simple/smooth hyperplane
                }else{
                        complex hyperplane
                }
                
    '''
classifier=svm.SVC(gamma='auto')
classifier.fit(X,Y)
print(classifier.predict([[2.5,4.5]])) # blue class
print(classifier.predict([[6,6]]))# red class

