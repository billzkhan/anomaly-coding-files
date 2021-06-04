import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import seaborn as sns
from datetime import datetime

speed_1 = pd.read_csv('D:/BILAL/5. Flooding/daejon network master data/MRT_BASE_TF_INFO_5MN_20200730.csv')

#taking out one link speed w.r.t time
speed_1 = speed_1[speed_1['LINK_ID']==1830001905]
speed_1['time'] = speed_1['HH_ID'].astype(str)+':'+speed_1['MN_ID'].astype(str)
speed_1['YMD_ID'] = speed_1['YMD_ID'].astype(str)
speed_1['date'] = speed_1['YMD_ID']+ " "+ speed_1['time']
speed_1[['date']] = speed_1[['date']].apply(pd.to_datetime, format='%Y%m%d %H:%M:%S.%f')
df = speed_1[['date','TRVL_SPD']]

#Steps
# Train an LSTM autoencoder on the data. 
# Using the LSTM autoencoder to reconstruct the error on the test data from 2013–09–04 to 2020–09–03.
# If the reconstruction error for the test data is above the threshold, we label the data point as an anomaly.


sns.lineplot(x=df['date'], y=df['TRVL_SPD'])

print('Start date is:', df['date'].min())
print('End date is:', df['date'].max())

#train and test
train,test = df.loc[df['date'] <= '2020-07-30 18:40:00'], df.loc[df['date']> '2020-07-30 18:40:00']
#train plot
sns.lineplot(x=train['date'], y=train['TRVL_SPD'])
#test plot
sns.lineplot(x=test['date'], y=test['TRVL_SPD'])

scaler = StandardScaler()
scaler = scaler.fit(train[['TRVL_SPD']])

train['TRVL_SPD'] = scaler.transform(train[['TRVL_SPD']])
test['TRVL_SPD'] = scaler.transform(test[['TRVL_SPD']])

seq_size = 20 #number of time steps to look back, larger sequences may improve forecasting

def to_sequences(x,y,seq_size=1):
    x_values = []
    y_values = []
    
    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values),np.array(y_values)

trainX, trainY = to_sequences(train[['TRVL_SPD']], train['TRVL_SPD'], seq_size)
testX, testY = to_sequences(test[['TRVL_SPD']], test['TRVL_SPD'], seq_size)

#modeling
model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(trainX.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(trainX.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()

#fit model
history = model.fit(trainX,trainY, epochs=500,batch_size=32, validation_split=0.1,verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'],label='Validation loss')
plt.legend()

#anomaly is where reconstruction error is large
trainPredict = model.predict(trainX)
trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)
plt.hist(trainMAE, bins=30)
max_trainMAE = 0.5 #or define 90% value of max as threshold

testPredict = model.predict(testX)
testMAE = np.mean(np.abs(testPredict - testX), axis=1)
plt.hist(testMAE, bins=30)

#Capture all details in a DataFrame for easy plotting
anomaly_df = pd.DataFrame(test[seq_size:])
anomaly_df['testMAE'] = testMAE
anomaly_df['max_trainMAE'] = max_trainMAE
anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
anomaly_df['TRVL_SPD'] = test[seq_size:]['TRVL_SPD']

#Plot testMAE vs max_trainMAE
sns.lineplot(x=anomaly_df['date'], y=anomaly_df['testMAE'])
sns.lineplot(x=anomaly_df['date'], y=anomaly_df['max_trainMAE'])

anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

#Plot anomalies
sns.lineplot(x=anomaly_df['date'], y=scaler.inverse_transform(anomaly_df['TRVL_SPD']))
sns.scatterplot(x=anomalies['date'], y=scaler.inverse_transform(anomalies['TRVL_SPD']), color='r')









