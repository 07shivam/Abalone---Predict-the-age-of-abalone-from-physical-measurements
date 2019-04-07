# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:45:54 2019

@author: Shivam Bhargava
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading dataset
data=pd.read_csv("abalone_csv.csv")
x=data.iloc[:,:-1]
data['age']=data['Class_number_of_rings']+1.5  
data.drop('Class_number_of_rings',axis=1)
y=data.iloc[:,-1]

print("Printing x \n",x)
print("printing y \n",y)
print("printing unique values \n",np.unique(y))

print('Number of attritube',x.shape[1])
print('Number of instance',x.shape[0])

#Encoding label sex
from sklearn.preprocessing import LabelEncoder
labl=LabelEncoder()
x['Sex']=labl.fit_transform(x['Sex'])

#standardizing
from sklearn.preprocessing import StandardScaler
standardScale=StandardScaler()
x=standardScale.fit_transform(x)

#splitting of data in 25-75%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=None)
print("Data Head \n",data.head())
print("Data train shape \n",x_train.shape)
print("Data test shape \n",x_test.shape)

#plotting data in the graph form, #Visualisation of dataset
data.hist(figsize=(20,10),grid=False,layout=(2,6),bins=40)
plt.show()    

#Fitting random forest
from sklearn.ensemble import RandomForestRegressor
regr=RandomForestRegressor(max_depth=2,random_state=0,n_estimators=3)
regr.fit(x_train,y_train)
y_pred=regr.predict(x_test)
#y_demo=list(y_pred)

#Calculating mean square error.
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score
rms=sqrt(mean_squared_error(y_test,y_pred))
print("Accuracy : ",100 - rms)

#Calculating Best Estimator from range 1 to 100
error=1000
error_index=0
for i in range(1,100):
    regr=RandomForestRegressor(max_depth=2, random_state=0, n_estimators=i)
    regr.fit(x_train,y_train)
    y_pred=regr.predict(x_test)
    #y_demo=list(y_pred)
    rms=sqrt(mean_squared_error(y_test,y_pred))
    if rms<error:
        error=rms
        error_index=i

#print best value of estimator
print("Best value of estimator is ", error_index)