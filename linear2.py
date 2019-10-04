# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 05:38:04 2019

@author: user
"""
#Housing DATASET
#LINEAR REGRESSION

import numpy as np  #numeric python library used for mathematical tools
import pandas as pd #for storing the huge datasets
import matplotlib.pyplot as plt #for visual representation 


#importing dataset
dataset = pd.read_csv("D:\MCA\MCA 5 SEM\ml\pro\Linear Regression\Housing_Data.csv")

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

print(X)
print(y)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.0180,random_state=0)


#we dont need preprocessing bcz inbuilt linear rgression contains it..

#fitting the test set results
from sklearn.linear_model import LinearRegression
regressor=LinearRegression() #regressor is object LinearRegression() is constructor
regressor.fit(X_train,y_train)


# predicting the test results
y_pred=regressor.predict(X_test)

plt.scatter(X_train, y_train, color ='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Area vs Pricess(training set)')
plt.xlabel('Area')
plt.ylabel('Prices')
plt.show()


#predicting the Test results
plt.scatter(X_test, y_test, color ='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Area vs Pricess(training set)')
plt.xlabel('Area')
plt.ylabel('Prices')
plt.show()



from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)




df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})











