#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[25]:


dt_train=pd.read_csv(r"D:\Python class\LSTM STOCK\Google_Stock_Price_Train.csv", date_parser = True)


# In[26]:


dt_train.head()


# ## Using Open Stock Price To Train Model

# In[27]:


train_set= dt_train.iloc[:,1:2].values
print(train_set)


# In[28]:


print(train_set.shape)


# In[29]:


train_set


# 
# # Normalizing Dataset

# In[31]:


from sklearn.preprocessing import MinMaxScaler
scaled_train_set=scaler.fit_transform(train_set)
scaled_train_set


# # Creating X_Train & Y_Train

# In[32]:


X_train=[]
y_train=[]
for i in range (60,1258):
    X_train.append(scaled_train_set[i-60:i, 0])
    y_train.append(scaled_train_set[i,0])
X_train=np.array(X_train)
y_train=np.array(y_train)
    


# In[33]:


print(X_train.shape)
print(y_train.shape)


# In[10]:


X_train


# In[35]:


X_train.shape


# In[34]:


y_train


# ## Reshaping Data

# In[37]:


X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_train.shape


# ## Building Model

# In[38]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


# In[39]:


regressior = Sequential()

regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 120, activation = 'relu'))
regressior.add(Dropout(0.2))

regressior.add(Dense(units = 1))


# In[40]:


regressior .summary()


# # Fit the Model

# In[41]:


regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs=50, batch_size=32)


# ## Extracting the Actual Prices of Jan 2017

# In[42]:


dt_test=pd.read_csv(r"D:\Python class\LSTM STOCK\Google_Stock_Price_Test.csv")


# In[43]:


dt_test.head()


# In[44]:


actual_stock_price=dt_test.iloc[:,1:2].values


# In[45]:


actual_stock_price


# ## Preparing Input Model

# In[63]:


dt_total=pd.concat((dt_train["Open"], dt_test["Open"]), ignore_index = True)


# In[64]:


dt_total.tail()


# In[66]:


dt_total


# In[67]:


inputs=dt_total[len(dt_total)-len(dt_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)


# In[68]:


X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


# In[69]:


X_test.shape


# # Predicting the Values for Jan 2017 Stock Prices

# In[70]:


predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


# In[71]:


plt.plot(actual_stock_price, color='red', label='Actual Google Stock Price')
plt.plot(predicted_stock_price, color='green',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()


# In[ ]:




