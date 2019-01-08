# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 13:59:32 2018

@author: AAYUSH
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

#sigmoid function:
# p(i)=1/1+ exp[- (b0 + b1*x)]

# logistic regression handles classification datasets perfectly 
# ouput b/w 0 and 1
# gives out the probability 

x1 = np.array([0,0.6,1.1,1.5,1.8,2.5,3,3.1,3.9,4,4.9,5,5.1])
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

x2 = np.array([3,3.8,4.4,5.2,5.5,6.5,6,6.1,6.9,7,7.9,8,8.1])
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

X = np.array([[0],[0.6],[1.1],[1.5],[1.8],[2.5],[3],[3.1],[3.9],[4],[4.9],[5],[5.1],[3],[3.8],[4.4],[5.2],[5.5],[6.5],[6],[6.1],[6.9],[7],[7.9],[8],[8.1]])
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])

plt.plot(x1,y1,'ro',color='blue')
plt.plot(x2,y2,'ro',color='red')

# scikit learn will use "Maximum-Liklihood"--max liklihood function= log L(b,x)
model = LogisticRegression()
model.fit(X,y)

print("b0 is:", model.intercept_)
print("b1 is:", model.coef_)

#forming and printing the sigmoid function
def logistic(classifier, x):
	return 1/(1+np.exp(-(model.intercept_ + model.coef_ * x)))
	
for i in range(1,120):
	plt.plot(i/10.0-2,logistic(model,i/10.0),'ro',color='green')
    
    
plt.axis([-2,10,-0.5,2])
plt.show()

# if X is 10 then prediction prob will give [0.053, 0.94] meaning 
# that it ha 0.05 probablity of not paying(blue dots) and 0.94 probabilty of paying(red dots)
pred = model.predict_proba([[10]])
print("Prediction: ", pred)