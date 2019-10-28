
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras


# In[2]:


import numpy as np
import pandas as pd
from keras.utils import normalize


# In[3]:


dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")


# In[4]:


dataset_path


# In[18]:


column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration,','Year','Origin']


# In[60]:


raw_data = pd.read_csv(dataset_path, names=column_names, na_values='?', sep=' ', comment='\t', skipinitialspace=True)


# In[61]:


raw_data


# In[62]:


df = raw_data.copy()


# In[63]:


df.tail(10)


# In[64]:


df.isna()


# In[65]:


df.isna().sum()


# In[66]:


df.dropna(inplace=True)


# In[67]:


df.shape


# In[68]:


df.columns


# In[69]:


df['Origin']


# In[70]:


origin_country_names = ['USA','Europe','Japan']
# del df['Origin']


# In[71]:


df.columns


# In[72]:


origin = df['Origin']
origin


# In[73]:


def map_countries_to_orign_values(df, column_name, origin_value, origin=origin):
    df[column_name]=(origin == int(origin_value))*1.0


# In[74]:


map_countries_to_orign_values(df, 'USA', 1)


# In[75]:


df.columns


# In[76]:


map_countries_to_orign_values(df, 'Europe', 2)
map_countries_to_orign_values(df, 'Japan', 3)


# In[77]:


df.columns


# In[79]:


del df['Origin']
df.tail(5)


# In[80]:


df.columns


# In[114]:


def split_data(df=df, split=0.1):
    
    total_size = df.shape[0]
    print("Total size  : ", total_size)
    test_size = int(total_size*split)
    print("Split Size : ",test_size)
    train, test = pd.DataFrame(), pd.DataFrame()
    train = df.iloc[:-test_size,:]
    test = df.iloc[:test_size:,:]
    
    print("Train data shape : ", train.shape)
    print("Test data shape : ", test.shape)
    
    return train, test


# In[115]:


train, test = split_data(split=0.2)


# In[116]:


import seaborn as sns


# In[117]:


type(train)


# In[118]:


sns.pairplot(train[['MPG','Cylinders','Displacement','Weight']], diag_kind='kde')


# In[119]:


train


# In[120]:


y_tr, y_te = train.pop('MPG'), test.pop('MPG')


# In[124]:


x_tr, x_te = train, test


# In[134]:


tr_stats, te_stats = x_tr.describe(), x_te.describe()
tr_stats, te_stats = tr_stats.transpose(), te_stats.transpose()


# In[135]:


tr_stats


# In[136]:


def normalize(x, stats=tr_stats):
    return (x-stats['mean']/stats['std'])


# In[137]:


normalized_tr, normalized_te = normalize(x_tr), normalize(x_te)


# In[141]:


print"Input shape (test): ",len(normalized_te.keys())
print"Input shape (train): ",len(normalized_tr.keys())


# In[142]:


ip_shape = len(normalized_tr.keys())


# In[143]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


# In[169]:


import matplotlib.pyplot as plt


# In[183]:


model = Sequential()
model.add(Dense(768, input_shape=[ip_shape]))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(1)) #no need for activation since we are not splitting into categories.

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse'])


# In[184]:


model.summary()
history = model.fit(normalized_tr, y_tr, validation_data=(normalized_te, y_te), batch_size=32, epochs=1500)


# In[185]:


h = pd.DataFrame(history.history)
h['epoch']=history.epoch
h.head(5)


# In[186]:


def plot_history(history):
    h = pd.DataFrame(history.history)
    h['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epochs ==>')
    plt.ylabel('MAE (Label)')
    plt.plot(h['epoch'],h['mean_absolute_error'],label='Training Error')
    plt.plot(h['epoch'],h['val_mean_absolute_error'],label='Validation Error')
    plt.ylim([0,5])
    plt.legend()
    
    plt.figure()
    plt.xlabel('Epochs ==>')
    plt.ylabel('MSE (Label - squared)')
    plt.plot(h['epoch'],h['mean_squared_error'],label='Training Error')
    plt.plot(h['epoch'],h['val_mean_squared_error'],label='Validation Error')
    plt.ylim([0,20])
    plt.legend()
    
    plt.show()


# In[187]:


plot_history(history)

