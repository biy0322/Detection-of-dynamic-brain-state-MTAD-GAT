import os
import warnings

import numpy as np
import pandas as pd
import sranodec as anom
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def preprocess(generate=False, group='1-1', val_split=0.3, path='./data/SMD/', sr=True):
    if generate:
        names = np.sort(os.listdir(path + 'train'))
        for name in names:
            wpath = path + 'preprocess/' + name.split('.')[0]
            if not os.path.exists(wpath):
                os.makedirs(wpath)

            # read data
            train_x = pd.read_csv(path + 'train/' + name, header=None)
            test_x = pd.read_csv(path + 'test/' + name, header=None)
            test_y = pd.read_csv(path + 'test_label/' + name, header=None, names=['label'])

            # data scaling(normalizing)
            scaler = MinMaxScaler()
            scaler.fit(train_x.values)
            train_x[train_x.columns] = scaler.transform(train_x.values)
            test_x[test_x.columns] = scaler.transform(test_x.values)

            # data cleaning: 
            if sr:
                for col in train_x.columns:
                    spec = anom.Silency(24, 24, 100)
                    score = spec.generate_anomaly_score(train_x[col].values)
                    train_x[col].values[score > np.percentile(score, 99)] = np.NaN

            # interploate
            train_x = train_x.interpolate(method='linear')
            test_x = test_x.interpolate(method='linear')

            # 모든 객체가 0인 열은 제외
            cols = ~(train_x == 0).all(axis=0)
            train_x = train_x.loc[:, cols]
            test_x = test_x.loc[:, cols]

            # after data preprocessing => save
            train_x.to_csv(wpath + '/' + name.split('.')[0] + '_train.csv', index=False)
            test_x.to_csv(wpath + '/' + name.split('.')[0] + '_test.csv', index=False)
            test_y.to_csv(wpath + '/' + name.split('.')[0] + '_test_label.csv', index=False)
            cols.to_csv(wpath + '/' + name.split('.')[0] + '_columns.csv')

    train_x = pd.read_csv(path + 'preprocess/machine-' + group + '/machine-' + group + '_train.csv')
    test_x = pd.read_csv(path + 'preprocess/machine-' + group + '/machine-' + group + '_test.csv')
    test_y = pd.read_csv(path + 'preprocess/machine-' + group + '/machine-' + group + '_test_label.csv')

    # split train / valid dataset
    valid_x = train_x.iloc[-int(val_split * len(train_x)):, :]
    train_x = train_x.iloc[:-int(val_split * len(train_x)), :]

    return train_x.values, valid_x.values, test_x.values, test_y.values
