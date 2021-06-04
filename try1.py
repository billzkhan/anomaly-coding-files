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
import datetime

# speed_1 = pd.read_csv('D:/BILAL/5. shortest path/daejon network master data/MRT_BASE_TF_INFO_15MN_20200730.csv')
# speed_1 = speed_1[speed_1['LINK_ID']==1830001905]
# speed_1['Date'] = pd.to_datetime(speed_1['HH_ID'].astype(int).astype(str)+':'+speed_1['MN_ID'].astype(int).astype(str), format = '%H:%M').dt.time
# df = speed_1[['Date','TRVL_SPD']]
# df = df.rename(columns = {'TRVL_SPD': 'Close'}, inplace = False)

#Steps
# Train an LSTM autoencoder on the data. 
# Using the LSTM autoencoder to reconstruct the error on the test data from 2013–09–04 to 2020–09–03.
# If the reconstruction error for the test data is above the threshold, we label the data point as an anomaly.


dataframe = pd.read_csv('C:/Users/UserK/Downloads/GE.csv')
df = dataframe[['Date','Close']]
df['Date'] = pd.to_datetime(df['Date'])

sns.lineplot(x=df['Date'], y=df['Close'])

print('Start date is:', df['Date'].min())
print('End date is:', df['Date'].max())

#train and test
train,test = df.loc[df['Date'] <= '2020-12-30'], df.loc[df['Date']> '2020-12-30']
#train plot
sns.lineplot(x=train['Date'], y=train['Close'])
#test plot
sns.lineplot(x=test['Date'], y=test['Close'])

scaler = StandardScaler()
scaler = scaler.fit(train[['Close']])

train['Close'] = scaler.transform(train[['Close']])
test['Close'] = scaler.transform(test[['Close']])

seq_size = 30 #number of time steps to look back, larger sequences may improve forecasting

def to_sequences(x,y,seq_size=1):
    x_values = []
    y_values = []
    
    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values),np.array(y_values)

trainX, trainY = to_sequences(train[['Close']], train['Close'], seq_size)
testX, testY = to_sequences(test[['Close']], test['Close'], seq_size)

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
history = model.fit(trainX,trainY, epochs=200,batch_size=32, validation_split=0.1,verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'],label='Validation loss')
plt.legend()

#anomaly is where reconstruction error is large
trainPredict = model.predict(trainX)
trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)
plt.hist(trainMAE, bins=30)
max_trainMAE = 0.9 #or define 90% value of max as threshold

testPredict = model.predict(testX)
testMAE = np.mean(np.abs(testPredict - testX), axis=1)
plt.hist(testMAE, bins=30)

#Capture all details in a DataFrame for easy plotting
anomaly_df = pd.DataFrame(test[seq_size:])
anomaly_df['testMAE'] = testMAE
anomaly_df['max_trainMAE'] = max_trainMAE
anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
anomaly_df['Close'] = test[seq_size:]['Close']

#Plot testMAE vs max_trainMAE
sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['testMAE'])
sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['max_trainMAE'])

anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

#Plot anomalies
sns.lineplot(x=anomaly_df['Date'], y=scaler.inverse_transform(anomaly_df['Close']))
sns.scatterplot(x=anomalies['Date'], y=scaler.inverse_transform(anomalies['Close']), color='r')










