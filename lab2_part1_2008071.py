# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 00:15:17 2022

@author: Joud

"""
# 1-import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2-import dataSet
#mydataset = pd.read_csv("Salary_Data_Lab.csv")
mydataset= pd.read_csv("C:\\Users\\joudm\\Desktop\\ML\\Salary_Data_Lab.csv")


# 3-select varib
x = mydataset.iloc[:, :-1].values
y = mydataset.iloc[:, 1].values

# 4-splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )


# 5-training the module on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 6-predicting the test set result
y_pred = regressor.predict(X_test)


# 7-visualize the training set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color ='blue')
plt.title('(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



# 8-visualize the testSet result
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color ='blue')
plt.title('(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salarys')
plt.show()


# POLYREGRISION  


# 1-import dataSet
mydataset = pd.read_csv("C:\\Users\\joudm\\Desktop\\ML\\Position_Salaries.csv")

x= mydataset.iloc[:,1:2].values 

y= mydataset.iloc[:,2].values 


# 2-Fitting the Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x, y)

# 3-Visualizing the Linear Regression results
def viz_linear():

   plt.scatter(x, y, color='red')

   plt.plot(x, lin_reg.predict(x), color='blue')

   plt.title('Truth or Bluff (Linear Regression)')

   plt.xlabel('Position level')

   plt.ylabel('Salary')

   plt.show()

   return

viz_linear()













# 4-Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)

X_poly = poly_reg.fit_transform(x)

pol_reg = LinearRegression()

pol_reg.fit(X_poly, y)








# 5-Visualizing the Polymonial Regression results
def viz_polymonial():

   plt.scatter(x, y, color='red')

   plt.plot(x, pol_reg.predict(poly_reg.fit_transform(x)), color='blue')

   plt.title('Truth or Bluff (Linear Regression)')

   plt.xlabel('Position level')

   plt.ylabel('Salary')

   plt.show()

   return

viz_polymonial()







# 5-Predicting a new result with Linear Regression
lin_reg.predict([[5.5]])

#output should be 249500



# 6-Predicting a new result with Polymonial Regression
pol_reg.predict(poly_reg.fit_transform([[5.5]]))

#output should be 132148.43750003


















