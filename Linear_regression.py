# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 19:11:48 2018

@author: AAYUSH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

#reading the .csv into a dataframe
dataset=pd.read_csv("Z:\\Machine_Learning_Algorithms-Python\\Datasets\\kc_house_data.csv")

#Now since we are making a simple linear regression....hence we take only one feature
#we make predictions based on "one feature" i.e. sqft area

size_of_house=dataset['sqft_living'] #column name in the datset--X
price=dataset['price'] # column in the dataset---Y


#now because ML algorithms handle "Arrays" and not "Dataframes"
#so we have to convert "dataframes" to "arrays" with the help of numpy

x=np.array(size_of_house).reshape(-1,1)# reshape will exclude the array indexes
y=np.array(price).reshape(-1,1)

#getting the linear regression algorithm containing gradient descent logic 
model=LinearRegression()

#training the model using fit()
#the "gradient descent" algorithm is going to make optimizations
#will find optimal b0 and b1 for H(x) to minimize MSE
model.fit(x,y)

#getting the MSE and R value
regression_model_mse=mean_squared_error(x,y)
print("MSE:", math.sqrt(regression_model_mse))

#R value determines how strong the relation is b/w X and Y
#If R higher then accuracy of regression is more
print("R Squared Value:",model.score(x,y))


#now we can get optimal (global mininmum) "b0 (slope of line)" and "b1 (y intercept)"
print(model.coef_[0]) #b0
print(model.intercept_[0]) #b1


#visualize the dataset with the optmized trained and fitted model
# use matplotlib---matlab
plt.scatter(x,y,color='green')
plt.plot(x,model.predict(x),color='black')
plt.title("Linear Regression")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()


#predicting the prices
print("prediction by the model:",model.predict([[2000]])) #give the value of X in array format



