import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
#from pyemma import msm # not available on Kaggle Kernel
from sklearn.svm import OneClassSVM

# some function for later

# # return Series of distance between each point and his distance with the closest centroid
# def getDistanceByPoint(data, model):
#     distance = pd.Series()
#     for i in range(0,len(data)):
#         Xa = np.array(data.loc[i])
#         Xb = model.cluster_centers_[model.labels_[i]-1]
#         distance.set_value(i, np.linalg.norm(Xa-Xb))
#     return distance

# # train markov model to get transition matrix
# def getTransitionMatrix (df):
# 	df = np.array(df)
# 	model = msm.estimate_markov_model(df, 1)
# 	return model.transition_matrix

# def markovAnomaly(df, windows_size, threshold):
#     transition_matrix = getTransitionMatrix(df)
#     real_threshold = threshold**windows_size
#     df_anomaly = []
#     for j in range(0, len(df)):
#         if (j < windows_size):
#             df_anomaly.append(0)
#         else:
#             sequence = df[j-windows_size:j]
#             sequence = sequence.reset_index(drop=True)
#             df_anomaly.append(anomalyElement(sequence, real_threshold, transition_matrix))
#     return df_anomaly

speed_1 = pd.read_csv('D:/BILAL/5. Flooding/daejon network master data/MRT_BASE_TF_INFO_5MN_20200730.csv')

#taking out one link speed w.r.t time
speed_1 = speed_1[speed_1['LINK_ID']==1830001905]
speed_1['time'] = speed_1['HH_ID'].astype(str)+':'+speed_1['MN_ID'].astype(str)
speed_1['YMD_ID'] = speed_1['YMD_ID'].astype(str)
speed_1['date'] = speed_1['YMD_ID']+ " "+ speed_1['time']
speed_1[['date']] = speed_1[['date']].apply(pd.to_datetime, format='%Y%m%d %H:%M:%S.%f')
df = speed_1[['date','TRVL_SPD']]

# df.to_csv('D:/BILAL/link_speed.csv')

#understanding data
df['TRVL_SPD'].describe()
plt.scatter(range(df.shape[0]), np.sort(df['TRVL_SPD'].values))
plt.xlabel('index')
plt.ylabel('TRVL_SPD')
plt.title("TRVL_SPD distribution")
sns.despine()

sns.distplot(df['TRVL_SPD'])
plt.title("Distribution of TRVL_SPD")
sns.despine()

print("Skewness: %f" % df['TRVL_SPD'].skew())
print("Kurtosis: %f" % df['TRVL_SPD'].kurt())

#using isolation forest
model = IsolationForest(n_estimators=100,max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(df[['TRVL_SPD']])
df['scores']=model.decision_function(df[['TRVL_SPD']])
df['anomaly']=model.predict(df[['TRVL_SPD']])
df.head(20)
bil = model.predict(df[['TRVL_SPD']]).tolist()

anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)

outliers_counter = len(df[df['TRVL_SPD'] < 15])
print("Accuracy percentage:", 100*list(df['anomaly']).count(-1)/(outliers_counter))



# isolation_forest.fit(df['TRVL_SPD'].values.reshape(-1, 1))
# xx = np.linspace(df['TRVL_SPD'].min(), df['TRVL_SPD'].max(), len(df)).reshape(-1,1)
# anomaly_score = isolation_forest.decision_function(xx)
# outlier = isolation_forest.predict(xx)

plt.figure(figsize=(10,4))
plt.plot(xx, anomaly_score, label='anomaly score')
plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                 where=outlier==-1, color='r', 
                 alpha=.4, label='outlier region')
plt.legend()
plt.ylabel('anomaly score')
plt.xlabel('Sales')
plt.show();

df.iloc[25]






