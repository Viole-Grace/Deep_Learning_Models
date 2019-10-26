
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import random


# In[3]:


from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout
from tensorflow import keras


# In[4]:


#import dataset
from tensorflow.keras.datasets import fashion_mnist


# In[8]:


f_mnist = fashion_mnist


# In[59]:


(x_tr, y_tr), (x_te, y_te) = f_mnist.load_data()


# In[60]:


x_tr, x_te = normalize(x_tr), normalize(x_te)


# In[61]:


x_tr.shape


# In[62]:


x_tr[0].shape


# In[63]:


x_te.shape


# In[64]:


from tensorflow.keras.layers import Flatten

model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[65]:


from tensorflow.keras import backend as K


# In[66]:


history=model.fit(x_tr, y_tr, validation_data = (x_te, y_te), batch_size=32, epochs=5)


# In[68]:


pred_model = model.predict(x_te)


# In[69]:


pred_model[0]


# In[70]:


print(np.argmax(pred_model[0]))

