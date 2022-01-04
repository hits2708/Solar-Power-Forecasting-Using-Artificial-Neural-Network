#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import array


# In[2]:


df = pd.read_csv('D:\\newdata2000to2001.csv')


# In[3]:


df['Time'][len(df)-1] 


# In[4]:


df.head()


# In[5]:


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    #print(X)
    #print(y)
    return (X,y)
 
# define input sequence
raw_seq = ['GHI']
# choose a number of time steps
n_steps = 24
# split into samples
(X, y) = split_sequence(df['GHI'], n_steps)
# summarize the data
for i in range(len(X)):
    print(X[i], y[i])


# In[6]:


plt.plot(df['Time'],df['GHI'])
plt.show()


# In[7]:


df[df['GHI']!=0].hist(column='GHI')


# In[8]:


X = np.array(X)   
newX = X.tolist()


# In[9]:


print(newX[81])


# In[10]:


y = np.array(y).tolist()


# In[11]:


print(y[81])


# In[12]:


import tensorflow as tf
from tensorflow import keras
# define model
model = keras.Sequential([
    keras.layers.Dense(1, input_dim=n_steps, activation = 'relu') 
])

model.compile(optimizer='adam',   
              loss='mse')


# In[13]:


model.fit(newX, y, epochs=20, verbose=0) 


# In[14]:


X_test=df['GHI'][24*12:350]


# In[15]:


len(newX)


# In[16]:


def predictHR(yer,m,d,h):
    timeStamp = pd.Timestamp(year = yer,month=m,day=d,hour=h)
    timeStamp = str(timeStamp)
    i = 0
    while i <len(df):
        Time = df['Time'][i]
        if Time==timeStamp:
            test_index = i
            break
        i+=1
    print('Actual value:',y[test_index])
    X_test = newX[test_index]
    #y_act = y[test_index]
    X_test = np.reshape(np.array(X_test),(1,24))
    return model.predict(X_test)[0][0]


# In[17]:


print('Predicted:',predictHR(2000,12,8,9))


# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(newX,y,test_size=0.2)


# In[19]:


y_model = model.predict(newX)


# In[20]:


score = model.evaluate(X_test,y_test)
print('mean: ',np.mean(y_model))
print('variance: ',score)
print('Standard Deviation: ',math.sqrt(score))


# In[21]:


xa = np.linspace(0,max(y),max(y))


# In[22]:


get_ipython().run_line_magic('matplotlib', 'qt5')


# In[23]:


plt.scatter(y,y_model)
plt.plot(xa,xa,'-r')
plt.ylabel('Predicted',fontsize=50)
plt.xlabel('Actual',fontsize=50)
plt.tick_params(axis='x', labelsize=35)
plt.tick_params(axis='y', labelsize=35)

plt.show()


# In[24]:


xb = np.linspace(1,17496,26256)


# In[25]:


plt.plot(xb,y,label='Actual')
plt.legend(loc=1, prop={'size': 35})
plt.plot(xb,y_model,label='Predicted')
plt.legend(loc=1, prop={'size': 35})

plt.ylabel('Global Horizontal Irradiance (GHI)',fontsize=45)
plt.xlabel('Time(hours)',fontsize=50)

plt.tick_params(axis='x', labelsize=35)
plt.tick_params(axis='y', labelsize=35)

plt.show()


# In[26]:


X_test = newX[-1]
X_test = np.reshape(np.array(X_test),(1,24))


# In[27]:


for i in range(24):
    yp = model.predict(X_test)
    #print(X_test)
    print(yp[0][0])   #Predicted value
    X_test = np.delete(X_test,0)
    X_test = np.append(X_test,yp[0][0])
    X_test = np.reshape(np.array(X_test),(1,24))


# In[28]:


len(newX[-1])


# In[29]:


plotY = [*newX[-3],*newX[-2],*newX[-1],*X_test[0]]


# In[30]:


plotX = np.linspace(1,24*4,24*4)


# In[31]:


plt.plot(plotX,plotY)
plt.axvline(x=24*3,color='r')
plt.ylabel('Global Horizontal Irradiance (GHI)',fontsize=45)
plt.xlabel('Time(hours)',fontsize=50)

plt.tick_params(axis='x', labelsize=35)
plt.tick_params(axis='y', labelsize=35)

plt.show()


# In[32]:


error=[]
for i in range(len(y)):
    error.append(abs(y[i]-y_model[i])) #error


# In[33]:


error=[]
for i in range(len(y)):
    error.append(abs(y[i]-y_model[i])/(y[i]+0.0001))# relative error this line


# In[34]:


plt.plot(xb,error)
plt.xlabel('Time',fontsize=55)
plt.ylabel('Error',fontsize=55)

plt.tick_params(axis='x', labelsize=35)
plt.tick_params(axis='y', labelsize=35)

plt.show()


# In[35]:


plt.semilogy(xb,error) #semilog scale
plt.xlabel('Time',fontsize=55)
plt.ylabel('Error',fontsize=55)

plt.tick_params(axis='x', labelsize=35)
plt.tick_params(axis='y', labelsize=35)

plt.show()

