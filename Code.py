# -*- coding: utf-8 -*-
"""
Spyder Editor

"""



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:22:34 2022

@author: mohamaddalati
"""

# Load library 
from sklearn.linear_model import LinearRegression
import pandas as pd 

# Import data file 
df = pd.read_csv('//ToyotaCorolla.csv') # load data directory 


# Construct variables (Age, KM, HP, Automatic, CC, Doors, Cylinders, Gears, Weight) as predictors 
X = df.iloc[:, 3:12]
y = df['Price']


# Before separating the data, standardize the predictors to ensure that each input variable has the same range,
# so the impact on the model is similar
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 
scaled_X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(scaled_X, columns = X.columns) 

# Separate the data into 65% training and 35% test and specify random_state = 662
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.35, random_state = 662)

# Run linear Regression on training set 
lm = LinearRegression()
model = lm.fit(X_train, y_train) #finds the line of best fit 

# Task1. Generate the prediction value of the test set
y_test_pred = model.predict(X_test)
# Calculate MSE 
from sklearn.metrics import mean_squared_error
lm_mse_Test = mean_squared_error(y_test, y_test_pred)
print("MSE is:", round(lm_mse_Test))


## Task2. RIDGE REGRESSION 

# load libraries 
from sklearn.linear_model import Ridge 

# Run Ridge Regression with alpha or lambda = 1 based on the training dataset  
ridge1 = Ridge(alpha = 1)
model1 = ridge1.fit(X_train, y_train) #line of best fit for Ridge Regression

# Generate the prediction of the target value of the test dataset and find the MSE 
y_test_pred1 = model1.predict(X_test) 

# Calculate the MSE 
ridge_penalty_mse = mean_squared_error(y_test, y_test_pred1)
print("RR MSE is:", round(ridge_penalty_mse) )



## Task3. LASSO Model with alpha = 1
from sklearn.linear_model import Lasso 
lasso1 = Lasso(alpha=1)
model2 = lasso1.fit(X_train, y_train) #line of best fit for LASSO Regression 

# Predict the target value of the test dataset and find MSE 
y_test_pred2 = model2.predict(X_test) 

#Calculate MSE for LASSO 
lasso_mse = mean_squared_error(y_test, y_test_pred2)
print("LASSO MSE is:", round(lasso_mse))


## Task4. Ridge first
# load libraries 
from sklearn.linear_model import Ridge 

# Run RR with alpha or lambda = 10 based on the training dataset  
ridge10 = Ridge(alpha = 10)
model10 = ridge10.fit(X_train, y_train) #line of best fit for Ridge Regression

# Generate the prediction of the target value of the test dataset and find the MSE 
y_test_pred10 = model10.predict(X_test) 

# Calculate the MSE 
ridge10_penalty_mse = mean_squared_error(y_test, y_test_pred10)
print("RR10 MSE is:", round(ridge10_penalty_mse) )

##                                  Alpha = 100 
ridge100 = Ridge(alpha = 100)
model100 = ridge100.fit(X_train, y_train) #line of best fit for Ridge Regression

# Generate the prediction of the target value of the test dataset and find the MSE 
y_test_pred100 = model100.predict(X_test)

# Calculate the MSE 
ridge100_penalty_mse = mean_squared_error(y_test, y_test_pred100)
print("RR100 MSE is:", round(ridge100_penalty_mse) )

##                                  Alpha = 1000 
ridge1000 = Ridge(alpha = 1000)
model1000 = ridge1000.fit(X_train, y_train) #line of best fit for Ridge Regression

# Generate the prediction of the target value of the test dataset and find the MSE 
y_test_pred1000 = model1000.predict(X_test) 

# Calculate the MSE 
ridge1000_penalty_mse = mean_squared_error(y_test, y_test_pred1000)
print("RR1000 MSE is:", round(ridge1000_penalty_mse) )

##                                  Alpha = 10000 
ridge10000 = Ridge(alpha = 10000)
model10000 = ridge10000.fit(X_train, y_train) #line of best fit for Ridge Regression

# Generate the prediction of the target value of the test dataset and find the MSE 
y_test_pred10000 = model10000.predict(X_test) 

# Calculate the MSE 
ridge10000_penalty_mse = mean_squared_error(y_test, y_test_pred10000)
print("RR10000 MSE is:", round(ridge10000_penalty_mse) )

#Coefficients 
model10000.coef_
#All of the coefficients are zero, so we can conclude that all coefficients are not useful 



## LASSO 
#                       Alpha = 10
from sklearn.linear_model import Lasso 
lasso10 = Lasso(alpha=10)
model10 = lasso10.fit(X_train, y_train) #line of best fit for LASSO 

# Predict the target value of the test dataset and find MSE 
y_test_pred10 = model10.predict(X_test) 

#Calculate MSE for LASSO 
lasso_mse10 = mean_squared_error(y_test, y_test_pred10)
print("LASSO MSE is:", round(lasso_mse10))

#                       Alpha = 100
from sklearn.linear_model import Lasso 
lasso100 = Lasso(alpha=100)
model100 = lasso100.fit(X_train, y_train) #line of best fit for LASSO 

# Predict the target value of the test dataset and find MSE 
y_test_pred100 = model100.predict(X_test) 

#Calculate MSE for LASSO 
lasso_mse100 = mean_squared_error(y_test, y_test_pred100)
print("LASSO MSE is:", round(lasso_mse100))

#                       Alpha = 1000
from sklearn.linear_model import Lasso 
lasso1000 = Lasso(alpha=1000)
model1000 = lasso1000.fit(X_train, y_train) #line of best fit for LASSO 

# Predict the target value of the test dataset and find MSE 
y_test_pred1000 = model1000.predict(X_test) 

#Calculate MSE for LASSO 
lasso_mse1000 = mean_squared_error(y_test, y_test_pred1000)
print("LASSO MSE is:", round(lasso_mse1000))

#                       Alpha = 10000
from sklearn.linear_model import Lasso 
lasso10000 = Lasso(alpha=10000)
lassomodel10000 = lasso10000.fit(X_train, y_train) #line of best fit for LASSO 

# Predict the target value of the test dataset and find MSE 
y_test_pred10000 = lassomodel10000.predict(X_test) 

#Calculate MSE for LASSO 
lasso_mse10000 = mean_squared_error(y_test, y_test_pred10000)
print("LASSO MSE is:", round(lasso_mse10000))

#Coefficients:
lassomodel10000.coef_

# Findings:
# Linear Regression (LR) was the best model to predict the price of cars since it has the minimum MSE 
# LASSO Regression performs best since most predictors were useful




