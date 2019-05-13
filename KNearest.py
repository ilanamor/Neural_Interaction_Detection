import math
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error, precision_recall_fscore_support, f1_score
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
np.random.seed(0)

# params ****************
# file_path = r'C:\Users\karin\PycharmProjects\current\datasets\bike\hour_new2.csv'
file_path = r'C:\Users\karin\PycharmProjects\current\datasets\higgs\higgs.csv'
header=True
index=True
is_classification = True
is_Binary_classification = True
k_fold = 10
selected = (1,5)
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
features = []
for feature in selected:
    features.append(np.expand_dims(X[:, feature - 1], 1))
X=np.concatenate(features, 1)
auc = 0
fpr_avg =0
tpr_avg =0
f1_avg_score = 0
kfold = KFold(n_splits=k_fold, random_state=1992, shuffle=False)
for train, test in kfold.split(X):
    X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
    if not is_classification:
        scaler_y = StandardScaler()
        scaler_y.fit(y_train.reshape(-1, 1))
        y_train, y_test = scaler_y.transform(y_train.reshape(-1, 1)), scaler_y.transform(y_test.reshape(-1, 1))
    if is_classification:
        model =KNeighborsClassifier(n_neighbors=5)
        y_predict = model.fit(X_train, y_train).predict(X_test)
        if is_Binary_classification:
            auc = auc + roc_auc_score(y_test, y_predict)
            # print(confusion_matrix(y_test, y_predict))
            fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=2)
            fpr_avg = fpr_avg + fpr
            tpr_avg = tpr_avg + tpr
        else:
            f1_avg_score = f1_avg_score + (f1_score(y_test, y_predict, average='macro'))
    else:
        model = KNeighborsRegressor(n_neighbors=5)
        y_predict = model.fit(X_train, y_train).predict(X_test)
        auc = auc + math.sqrt(mean_squared_error(y_test,y_predict))
print(auc/k_fold)
if is_classification:
    print ("fpr avg =" ,(fpr_avg/k_fold) ," tpr avg =" ,(tpr_avg/k_fold) )
    if not is_Binary_classification:
        print(precision_recall_fscore_support(y_test, y_predict, average='macro'))
        print("f1 score:" ,f1_avg_score/k_fold)


