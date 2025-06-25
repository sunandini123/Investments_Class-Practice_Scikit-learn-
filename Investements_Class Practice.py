
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


dataset = pd.read_csv(r'/Users/shashi/Desktop/Investment.csv')

dataset

X=dataset.iloc[:,:-1]

y=dataset.iloc[:,4]


X=pd.get_dummies(X,dtype=int)


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0 )


# creates linear model line y=mx+c , where m is slope which says about increase in salary  and c is intercept denotes the basic salary 

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)


y_pred=regressor.predict(X_test)


bias = regressor.score(X_train, y_train)
print(bias)


variance=regressor.score(X_test, y_test)
print(variance)

intercept=regressor.intercept_
print(intercept)


X = np.append(arr = np.ones((50,1)).astype(int),values = X , axis =1)
# 1 is added, axis 1 means column 



import statsmodels.api as sm 
X_opt = X[:,[0,1,2,3,4,5]]
#ordinary least squares 
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# we are giving  1,2,3,4,5 values for X1,X2,X3,X4,X5 


import statsmodels.api as sm 
X_opt = X[:,[0,1,2,3]]
#ordinary least squares 
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# we are giving  1,2,3 for X1,X2,X3

import statsmodels.api as sm 
X_opt = X[:,[0,1]]
#ordinary least squares 
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
# we are giving  1,2,3,4,5 values for X1.

##  p-values — You’re checking them with regressor_OLS.summary(). If many predictors have high #p-values (> 0.05), it means those variables don’t add much predictive power.
## R² / Adjusted R² — The summary gives these. Adjusted R² is more reliable because it penalizes #for too many features.
##  bias / variance (your score on train & test) — If these are close and both reasonably high, your model is generalizing well → the data fits well.

































