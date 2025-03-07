import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests
import io

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
res = requests.get(url).content

df_auto = pd.read_csv(io.StringIO(res.decode("utf-8")), header=None)

df_auto.columns = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]

# print(df_auto.head())

df_auto = df_auto[['horsepower', 'width', 'height', 'price']]

# print(df_auto.isin(['?']).sum())

# 欠損値を含む行を削除
df_auto = df_auto.replace('?', np.nan).dropna()

df_auto = df_auto.assign(price=pd.to_numeric(df_auto['price']))
df_auto = df_auto.assign(horsepower=pd.to_numeric(df_auto['horsepower']))

# 多重共線性(説明変数同士に強い相関があること)を確認
# 回帰係数の分散が大きくなり、係数の有意性が損なわれてしまうため、相関の高い変数群からは代表して一つを選択して説明変数に加えるなどで対策する
print(df_auto.corr())

X = df_auto.drop('price', axis=1)
y = df_auto['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

print('決定係数(train):{:.3f}'.format(model.score(X_train, y_train)))
print('決定係数(test):{:.3f}'.format(model.score(X_test, y_test)))

print('\n回帰係数\n(train):{}'.format(pd.Series(model.coef_, index=X.columns)))
print('\n切片(train):{:.3f}'.format(model.intercept_))






