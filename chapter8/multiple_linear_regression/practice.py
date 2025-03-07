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

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# with, engine-size 5:5

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

df_auto = df_auto[['engine-size', 'width', 'price']]
# print(df_auto.isin(['?']).sum())
df_auto = df_auto.replace('?', np.nan).dropna()
# print(df_auto.dtypes)

df_auto = df_auto.assign(price=pd.to_numeric(df_auto['price']))


print(df_auto.corr())

X = df_auto.drop('price', axis=1)
y = df_auto['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)

print('決定係数(train):{:.3f}'.format(pipeline.score(X_train, y_train)))
print('決定係数(test):{:.3f}'.format(pipeline.score(X_test, y_test)))

model = pipeline.named_steps['model']
print('\n回帰係数\n(train):{}'.format(pd.Series(model.coef_, index=X.columns)))
print('\n切片(train):{:.3f}'.format(model.intercept_))



