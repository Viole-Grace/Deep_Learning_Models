import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras.utils import normalize

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration,','Year','Origin']

raw_data = pd.read_csv(dataset_path, names=column_names, na_values='?', sep=' ', comment='\t', skipinitialspace=True)

df = raw_data.copy()

df.tail(10)

df.isna()
df.isna().sum()

df.dropna(inplace=True)
df.shape

df.columns

df['Origin']

origin_country_names = ['USA','Europe','Japan']
df.columns

origin = df['Origin']
origin

def map_countries_to_orign_values(df, column_name, origin_value, origin=origin):
    df[column_name]=(origin == int(origin_value))*1.0

map_countries_to_orign_values(df, 'USA', 1)
df.columns

map_countries_to_orign_values(df, 'Europe', 2)
map_countries_to_orign_values(df, 'Japan', 3)
df.columns

del df['Origin']
df.tail(5)

df.columns

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

train, test = split_data(split=0.2)

import seaborn as sns

type(train)

sns.pairplot(train[['MPG','Cylinders','Displacement','Weight']], diag_kind='kde')

train

y_tr, y_te = train.pop('MPG'), test.pop('MPG')
x_tr, x_te = train, test

tr_stats, te_stats = x_tr.describe(), x_te.describe()
tr_stats, te_stats = tr_stats.transpose(), te_stats.transpose()

tr_stats

def normalize(x, stats=tr_stats):
    return (x-stats['mean']/stats['std'])

normalized_tr, normalized_te = normalize(x_tr), normalize(x_te)
print"Input shape (test): ",len(normalized_te.keys())
print"Input shape (train): ",len(normalized_tr.keys())

ip_shape = len(normalized_tr.keys())
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import matplotlib.pyplot as plt

model = Sequential()
model.add(Dense(1024, input_shape=[ip_shape]))
model.add(Activation('relu'))
model.add(Dense(768))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(1)) #no need for activation since we are not splitting into categories.

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse'])

model.summary()
history = model.fit(normalized_tr, y_tr, validation_data=(normalized_te, y_te), batch_size=256, epochs=1000)

h = pd.DataFrame(history.history)
h['epoch']=history.epoch
h.head(5)

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

plot_history(history)

from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=10) #wait for 10 epochs to check for improvement
history = model.fit(normalized_tr, y_tr, validation_data = (normalized_te, y_te),
                   batch_size=256, epochs=1000, verbose=0, callbacks=[early_stop])
plot_history(history)

check_pred = model.predict(normalized_te).flatten()
plt.xlabel('True Values (MPG)')
plt.ylabel('Predicted Values (MPG)')
plt.scatter(y_te, check_pred)
plt.axis('equal'); plt.axis('square')

plt.xlim(0, plt.xlim()[1])
plt.xlim(0, plt.ylim()[1])
dummy = plt.plot([-100,100],[-100,100])

e = check_pred - y_te
plt.hist(e, bins=10)
plt.xlabel('Prediction')
plt.ylabel('Count')