import csv
import math
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.model_selection import KFold, permutation_test_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score



def read_csv():
    df = pd.read_csv(file_path) if header else pd.read_csv(file_path, header=None)
    df = df.drop(df.columns[[0]], axis=1) if index else df  # without the index column

    range = df.shape[1] - 1 if is_classification else df.shape[1]
    for y in df.columns[0:range]:
        if (df[y].dtype == np.float32):
            df[y] = df[y].astype('float64')
        elif (df[y].dtype == np.int32):
            df[y] = df[y].astype('int64')
        else:
            continue
    return df

def prepare_data(train, test, X_full, Y_full, num_input):
    tr_x, te_x, tr_y, te_y = X_full[train], X_full[test], Y_full[train], Y_full[test]

    for j in selected:
        i=j-1
        if df[df.columns[i]].dtype == np.float64:
            scaler_x = StandardScaler()
            scaler_x.fit(tr_x.T[i].reshape(-1, 1))
            tr_x_tmp, te_x_tmp = scaler_x.transform(tr_x.T[i].reshape(-1, 1)), scaler_x.transform(te_x.T[i].reshape(-1, 1))
            tr_x.T[i]=tr_x_tmp.flatten()
            te_x.T[i]=te_x_tmp.flatten()
        elif df[df.columns[i]].dtype == np.object:
            label_encoder = LabelEncoder()
            label_encoder.fit_transform(tr_x.T[i].reshape(-1, 1))
            tr_x_tmp, te_x_tmp, va_x_tmp = label_encoder.transform(tr_x.T[i].reshape(-1, 1)), label_encoder.transform(
                te_x.T[i].reshape(-1, 1))
            tr_x.T[i] = tr_x_tmp.flatten()
            te_x.T[i] = te_x_tmp.flatten()

    if not is_classification:
        scaler_y = StandardScaler()
        scaler_y.fit(tr_y.reshape(-1, 1))
        tr_y, te_y = scaler_y.transform(tr_y.reshape(-1, 1)), scaler_y.transform(te_y.reshape(-1, 1))

    return tr_x, te_x, tr_y, te_y

def get_slice_of_data(selected, X_all):
    columns = [x-1 for x in selected]
    return X_all[:, columns]

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

# ******************************* PARAMS ***************************
file_path = r'C:\Users\Ilana\PycharmProjects\Neural_Interaction_Detection\datasets\higgs\higgs.csv'
header = True
index = True
is_classification = True
k_fold = 10
# selected = [13,15,8,9,5,10,3,6,11,12,16,1,4,7,14,2]
# selected = [14,11,7,12,10,6,9,16,13,3,8,1,2,15,5,4]
# selected = [5,7,2,6,4,9,10,8,11,1,12,3]
# selected = [4,9,10,11,3,2,1,7,12,6,8,5]
# selected = [1,2,5,6,4,3,8,7]
# selected = [1,8,2,4,3,7,5,6]
selected = [3,8,20,12,22,23,1,14,25,26,2,9,30,17,24,6,11,10,4,15,18,27,7,29,13,5,19,28,21,16]
# selected = [1,5,2,7,13,3,6,28,29,27,25,14,24,12,26,11,23,10,30,4,22,8,20,18,15,17,19,9,16,21]
exp_output = []
# ****************************************************************

for i in range(len(selected),0,-1):
    np.random.seed(0)
    df = read_csv()
    X_all = df.iloc[:, 0:df.shape[1] - 1].values
    y = df.values[:,-1]
    X = get_slice_of_data(selected, X_all)

    auc = 0
    acc = 0
    rmse = 0
    kfold = KFold(n_splits=k_fold, random_state=0, shuffle=False)
    for train, test in kfold.split(X):
        X_train, X_test, y_train, y_test = prepare_data(train, test, X, y, num_input=X.shape[1]-1)

        if is_classification:
            model = KNeighborsClassifier(n_neighbors=5)
            y_predict = model.fit(X_train, y_train).predict(X_test)
            auc += multiclass_roc_auc_score(y_test, y_predict, average="weighted")
            acc += accuracy_score(y_test, y_predict)
        else:
            model = KNeighborsRegressor(n_neighbors=5)
            y_predict = model.fit(X_train, y_train).predict(X_test)
            rmse += math.sqrt(mean_squared_error(y_test,y_predict))

    if is_classification:
        exp_output.insert(0,[i, acc / k_fold, auc/k_fold])
    else:
        exp_output.insert(0,[i, rmse / k_fold])

    del selected[-1]

with open("experiment_results_kn.csv", mode='w',newline='') as exp_file:
    if is_classification:
        exp_output.insert(0, ['K', 'acc', 'auc'])
    else:
        exp_output.insert(0, ['K', 'rmse'])
    wr = csv.writer(exp_file, dialect='excel')
    wr.writerows(exp_output)
