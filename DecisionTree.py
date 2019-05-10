import math
import sklearn.tree as tree
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# params ****************
file_path = r'C:\Users\karin\Desktop\cal_housing_changes.csv'
# file_path = r'C:\Users\karin\PycharmProjects\current\datasets\cal housing\cal_housing.csv'
header=True
index=False
is_classification = False
kfolds = 10
# ************************
df = pd.read_csv(file_path) if header else pd.read_csv(file_path, header=None)
df = df.drop(df.columns[[0]], axis=1) if index else df #without the index column

range = df.shape[1] - 1 if is_classification else df.shape[1]
for y in df.columns[0:range]:
        if (df[y].dtype == np.int32 or df[y].dtype == np.int64 or df[y].dtype == np.float32):
            df[y] = df[y].astype('float64')
        elif (df[y].dtype == np.object):
            label_encoder = LabelEncoder()
            df[y] = label_encoder.fit_transform(df[y]).astype('float64')
        else:
            continue
X = df.values[:,:-1,]
Y = df.values[:,-1]
auc = 0
i=0
while i < kfolds:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    if is_classification:
        model = tree.DecisionTreeClassifier()
        y_predict = model.fit(X_train, y_train).predict(X_test)
        auc = auc + roc_auc_score(y_test, y_predict)
    else:
        model = tree.DecisionTreeRegressor()
        y_predict = model.fit(X_train, y_train).predict(X_test)
        auc = auc + math.sqrt(mean_squared_error(y_test,y_predict))
    i=i+1
print(auc/kfolds)


