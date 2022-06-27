#!/usr/bin/env python
# coding: utf-8

# In[80]:

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# In[81]:

df = pd.read_csv('headbrain.csv')
df.head()


# In[82]:

# Collecting X & Y Values 
X = df['Head Size(cm^3)'].values
y = df['Brain Weight(grams)'].values


# In[98]:


from sklearn.linear_model import LinearRegression

# Reshape starting from 1
X = X.reshape(len(X), 1)

# Creating Model
reg = LinearRegression()

# Fitting Training Data 
reg = reg.fit(X, y)

# Y Predictions on x test
Y_pred = reg.predict(X)

# Graphing Line 
plt.plot(X, Y_pred, color='#58b970', label='Regression Line')

# Plotting Scattor Points
plt.scatter(X, y, color ='#ef5423', label='Scatter Plot')

# Label each axis 
plt.xlabel("Head Size(cm^3)")
plt.ylabel("Brain Weight(grams)")

# Show
plt.legend()
plt.show()


# In[100]:

# Calculate The Accuracy of Model 
from sklearn.metrics import mean_squared_error 

# R Squared Method preformed by reg.score(X, y) using mean_squared_error
accuracy = reg.score(X, y)
print("The accuracy is " + str(round(accuracy * 100)) + '%')
