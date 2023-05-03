# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 15:45:21 2023

@author: joudm
"""

# 1-import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2-import dataSet
mydataset= pd.read_csv("C:\\Users\\joudm\\Desktop\\ML\\Position_Salaries.csv")
X = mydataset.iloc[:, 1:2].values
y = mydataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y= np.ravel (sc_y.fit_transform(y.reshape(-1,1)))#considerit as an array and reshape the y as 2D vector

from sklearn.svm import SVR #it will add any kind of help
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

y_pred= sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)) #we will predict the value of y /we need 2D for SVR/ reshape it 
#visualize 
plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X),color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




