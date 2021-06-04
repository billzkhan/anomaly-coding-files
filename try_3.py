import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
%matplotlib inline

from numpy.random import seed
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

# seed(10)
# tf.random.set_seed(10)

speed_1 = pd.read_csv('D:/BILAL/5. Flooding/daejon network master data/MRT_BASE_TF_INFO_5MN_20200730.csv')

#taking out one link speed w.r.t time
speed_1 = speed_1[(speed_1['LINK_ID']==1830001905)]
#datetime editing
speed_1['time'] = speed_1['HH_ID'].astype(str)+':'+speed_1['MN_ID'].astype(str)
speed_1['YMD_ID'] = speed_1['YMD_ID'].astype(str)
speed_1['date'] = speed_1['YMD_ID']+ " "+ speed_1['time']
speed_1[['date']] = speed_1[['date']].apply(pd.to_datetime, format='%Y%m%d %H:%M:%S.%f')
#converting long to wide dataframe
speed_1 = speed_1.pivot(index='date', columns='LINK_ID', values='TRVL_SPD').reset_index()
speed_1.columns = ['date','link_1']
#final data
df = speed_1
df.to_csv("D:/BILAL/link_2.csv")

train,test = df.loc[df['date'] <= '2020-07-30 18:40:00'], df.loc[df['date']> '2020-07-30 18:40:00']
train.set_index('date', inplace=True)
test.set_index('date', inplace=True)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.fit_transform(test)
scaler_filename = 'scaler_data'
joblib.dump(scaler,scaler_filename)

X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

#define autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1],X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True,
              kernel_regularizer= regularizers.l2(0.00))(inputs)
    L2 = LSTM(4,activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation ='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation ='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model

model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()


#fit model
history = model.fit(X_train,X_train, epochs=100,batch_size=16, validation_split=0.05,verbose=1) #why used trainX two times

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'],label='Validation loss')
plt.legend()

#anomaly is where reconstruction error is large
x_pred = model.predict(X_train)
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[2])
x_pred = pd.DataFrame(x_pred, columns = train.columns)
x_pred.index = train.index

scored = pd.DataFrame(index = train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(x_pred-Xtrain),axis=1)
plt.figure(figsize=(16,9),dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins = 20, kde = True, color='blue')
plt.xlim([0.0,.5]) 

#calculate loss on test set
x_pred = model.predict(X_test)
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[2])
x_pred = pd.DataFrame(x_pred, columns = test.columns)
x_pred.index = test.index

scored = pd.DataFrame(index = test.index)
Xtest = X_test.reshape(X_test.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(x_pred-Xtest),axis=1)
scored['Threshold'] = 0.09
scored['Anomaly'] = scored['Loss_mae']> scored['Threshold']
scored.head()

#calculate same metrics for training set and mnerge all data in single data frame
x_pred_train = model.predict(X_train)
x_pred_train = x_pred_train.reshape(x_pred_train.shape[0],x_pred_train.shape[2])
x_pred_train = pd.DataFrame(x_pred_train, columns = train.columns)
x_pred_train.index = train.index

scored_train = pd.DataFrame(index = train.index)
scored_train['Loss_mae'] = np.mean(np.abs(x_pred_train-Xtrain),axis=1)
scored_train['Threshold'] = 0.09
scored_train['Anomaly'] = scored['Loss_mae']> scored['Threshold']
scored = pd.concat([scored_train,scored])    

scored.plot(logy = True, figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red'])

model.save("Bilal_anomaly.h5")


# import tabpy_client
# from tabpy.tabpy_tools.client import Client
# client = tabpy_client.Client('http://localhost:9004/')

# def anomaly_pred( _arg1):
#     import pandas as pd

#     #Get the new app's data in a dictionary
#     row = {'link_1': _arg1}
    
#     #Convert it into a dataframe
#     test_data = pd.DataFrame(data = row,index=[0])
#     #Predict the survival and death probabilities
#     pred_anom = model.predict(test_data)
#     x=2
#     #Return only the survival probability
#     # return pred_anom.tolist()
#     return x

# client.deploy('anomaly_pred', anomaly_pred,'Predicts anomaly_pred', override = True)

































