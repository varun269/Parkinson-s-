# Parkinson-s-
import numpy as np
import pandas as pd
import seaborn as sn
import gc
import matplotlib.pyplot as plt
from sklearn import *
import glob
path="/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/"

train = glob.glob(path+'train/defog/**')
train1 = glob.glob(path+'train/tdcsfog/**')
def get_data(f):
    dataframe = pd.read_csv(f)
    dataframe['Id'] = f.split('/')[-1].split('.')[0]
    dataframe['data_type'] = f.split('/')[-2]
    return dataframe
dataframe_train = pd.concat([get_data(f) for f in train])
dataframe_train1 = pd.concat([get_data(f) for f in train1])
dataframe_train=pd.concat([dataframe_train,dataframe_train1])
print(dataframe_train.shape)
print(dataframe_train1.shape)
print(dataframe_train.shape)
dataframe_train.fillna(0,inplace=True)
print(dataframe_train.isnull().sum().sum())
features=['Time', 'AccV', 'AccML', 'AccAP']
Targets=['StartHesitation', 'Turn' , 'Walking']
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(dataframe_train[features], dataframe_train[Targets], test_size=.30, random_state=42)
del dataframe_train
gc.collect()
model_Reg = ensemble.RandomForestRegressor(n_estimators=100, max_depth=7, n_jobs=-1, random_state=42)
model_Reg.fit(X_train, y_train)
print(metrics.average_precision_score(y_valid, model_Reg.predict(X_valid).clip(0.0,1.0)))
sub1 = pd.read_csv(path+'sample_submission.csv')
test = glob.glob(path+'test/**/**')

sub1['t'] = 0
submission = []
for f in test:
    dataframe = pd.read_csv(f)
    dataframe['Id'] = f.split('/')[-1].split('.')[0]
    dataframe = dataframe.fillna(0).reset_index(drop=True)
    res = pd.DataFrame(np.round(model_Reg.predict(dataframe[features]),3), columns=['StartHesitation', 'Turn' , 'Walking'])
    dataframe = pd.concat([dataframe,res], axis=1)
    dataframe['Id'] = dataframe['Id'].astype(str) + '_' + dataframe['Time'].astype(str)
    submission.append(dataframe[['Id','StartHesitation', 'Turn' , 'Walking']])
submission = pd.concat(submission)
submission = pd.merge(sub1[['Id','t']], submission, how='left', on='Id').fillna(0.0)
submission[['Id','StartHesitation', 'Turn' , 'Walking']].to_csv('submission.csv', index=False)
