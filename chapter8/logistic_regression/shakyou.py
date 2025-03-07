import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import io

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

res = requests.get(url).content

df_adult = pd.read_csv(io.StringIO(res.decode("utf-8")), header=None)

df_adult.columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'flg-50k'
]

print('データセットのサイズ:{}'.format(df_adult.shape))
print('欠損値の数:{}'.format(df_adult.isnull().sum()))

# flg-50kが [' <=50K', ' >50K'] の2値になっているため、これを [0, 1] に変換
df_adult['fin_flg'] = df_adult['flg-50k'].map(lambda x: 1 if x == ' >50K' else 0)

print(df_adult.groupby('fin_flg').size())

X = df_adult[['age', 'fnlwgt', 'capital-gain', 'capital-loss']]
y = df_adult['fin_flg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 標準化
scaler = StandardScaler()

scaler.fit(X_train)  # fit()だけを実行
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train_std, y_train)

print('正解率(train):{:.3f}'.format(model.score(X_train_std, y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test_std, y_test)))

print(model.coef_)


