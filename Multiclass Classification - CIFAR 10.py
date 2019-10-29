
# coding: utf-8

# In[18]:


import tensorflow as tf


# In[19]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, Dropout
from tensorflow.keras.utils import normalize
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# In[20]:


import numpy as np
import math


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[22]:


from tensorflow.keras.datasets import cifar10


# In[53]:


(x_tr, y_tr), (x_te, y_te) = cifar10.load_data()
x_tr, x_te = x_tr/255.0, x_te/255.0


# In[54]:


# x_tr, x_te = normalize(x_tr), normalize(x_te)
help(cifar10)


# In[55]:


class_names = ['aeroplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# In[56]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_tr[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_tr[i][0]])
plt.show()


# In[57]:


x_tr.shape


# In[58]:


from tensorflow.keras.layers import Dense


# In[59]:


model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[60]:


history=model.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=10)


# In[61]:


val_loss, val_acc = model.evaluate(x_te, y_te, verbose=2)


# In[63]:


history.history


# In[65]:


plt.plot(history.history['acc'], label='Training accuracy')
plt.plot(history.history['val_acc'], label='Testing accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend()
plt.show()

