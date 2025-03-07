from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataset = load_breast_cancer()

# print(dataset.data.shape)
# print(dataset.feature_names)

df_cancer = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# print(df_cancer.head())

# print(dataset.target.shape)
# print(dataset.target_names)

# print('\n--- 欠損値の確認 ---')
# print('欠損値の数:\n{}'.format(df_cancer.isnull().sum()))

# print('\n欠損値の割合:')
# print((df_cancer.isnull().sum() / len(df_cancer)).apply(lambda x: f'{x:.2%}'))

# print('\n--- 基本統計量 ---')
# print(df_cancer.describe())

# print('\n--- 異常値の確認 ---')
# print(df_cancer.apply(lambda x: np.sum(np.abs(x - x.mean()) > (3 * x.std()))))

X = df_cancer
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

scaler = StandardScaler()

scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_std, y_train)

# 0.968->0.989
print('正解率(train):{:.3f}'.format(model.score(X_train_std, y_train)))
# 0.954->0.975
print('正解率(test):{:.3f}'.format(model.score(X_test_std, y_test)))

print(model.coef_)









