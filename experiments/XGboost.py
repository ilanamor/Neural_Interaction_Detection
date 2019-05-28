import csv
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score


def read_csv():
    df = pd.read_csv(file_path) if header else pd.read_csv(file_path, header=None)
    df = df.drop(df.columns[[0]], axis=1) if index else df  # without the index column

    range = df.shape[1] - 1 if is_classification else df.shape[1]
    for y in df.columns[0:range]:
        if (df[y].dtype == np.float32):
            df[y] = df[y].astype('float64')
            scaler = StandardScaler()
            df[y] = scaler.fit_transform(df[y])
        elif (df[y].dtype == np.int32):
            df[y] = df[y].astype('int64')
        elif df[y].dtype == np.object:
            label_encoder = LabelEncoder()
            df[y] = label_encoder.fit_transform(df[y])
    return df

def prepare_data(train, test, X_full, Y_full, num_input):
    tr_x, te_x, tr_y, te_y = X_full[train], X_full[test], Y_full[train], Y_full[test]
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
file_path = r'C:\Users\Ilana\PycharmProjects\Neural_Interaction_Detection\datasets\Mice\Data_Cortex_Nuclear.csv'
header = True
index = False
is_classification = True
is_binary_classification = True
k_fold = 10
# selected = [13,15,8,9,5,10,3,6,11,12,16,1,4,7,14,2]
# selected = [14,11,7,12,10,6,9,16,13,3,8,1,2,15,5,4]
# selected = [5,7,2,6,4,9,10,8,11,1,12,3]
# selected = [4,9,10,11,3,2,1,7,12,6,8,5]
# selected = [1,2,5,6,4,3,8,7]
# selected = [1,8,2,4,3,7,5,6]
# selected = [19, 148, 118, 54, 9, 7, 143, 114, 78, 57, 166, 17, 40, 146, 16, 66, 81, 126, 52, 55, 86, 13, 26, 139, 5, 122, 6, 106, 138, 20, 58, 142, 43, 141, 60, 48, 68, 115, 104, 70, 131, 27, 79, 63, 140, 93, 108, 164, 46, 128, 31, 83, 111, 94, 25, 155, 65, 145, 125, 101, 2, 49, 62, 22, 110, 113, 134, 36, 92, 112, 144, 162, 29, 95, 154, 119, 137, 77, 14, 98, 121, 73, 136, 124, 53, 97, 150, 87, 165, 45, 35, 84, 129, 159, 100, 103, 42, 130, 50, 120, 135, 153, 41, 32, 109, 149, 24, 76, 123, 34, 158, 64, 96, 61, 80, 3, 51, 15, 10, 59, 133, 147, 12, 152, 102, 30, 8, 91, 88, 89, 163, 71, 18, 82, 56, 105, 151, 69, 4, 99, 156, 67, 72, 38, 107, 74, 23, 1, 167, 21, 47, 160, 132, 37, 28, 90, 157, 75, 44, 11, 33, 127, 117, 161, 85, 39, 116]
# selected = [151, 162, 92, 36, 166, 102, 163, 110, 3, 42, 109, 144, 68, 99, 79, 165, 95, 7, 126, 96, 160, 45, 153, 35, 131, 104, 156, 122, 14, 143, 111, 161, 66, 164, 154, 124, 71, 155, 72, 62, 132, 80, 98, 69, 89, 25, 32, 77, 29, 70, 100, 19, 78, 125, 74, 12, 105, 17, 23, 38, 4, 63, 135, 57, 123, 44, 65, 30, 148, 83, 119, 137, 112, 8, 118, 49, 103, 9, 59, 138, 149, 11, 113, 82, 75, 106, 58, 120, 97, 22, 114, 134, 55, 61, 15, 52, 85, 117, 84, 136, 1, 40, 87, 142, 39, 27, 26, 64, 20, 127, 86, 90, 33, 121, 101, 152, 48, 41, 73, 46, 130, 116, 159, 93, 115, 6, 139, 50, 150, 21, 108, 145, 13, 91, 2, 51, 47, 18, 53, 146, 81, 60, 37, 128, 28, 141, 24, 54, 56, 129, 67, 34, 107, 31, 94, 133, 158, 140, 43, 10, 88, 16, 5, 157, 76, 147]
# selected = [65, 77, 78, 60, 80, 72, 168, 39, 76, 121, 85, 102, 156, 1, 152, 79, 15, 53, 107, 74, 106, 75, 148, 145, 38, 56, 48, 133, 108, 16, 73, 150, 14, 25, 157, 59, 41, 40, 99, 69, 166, 116, 21, 84, 149, 55, 13, 12, 58, 146, 160, 167, 86, 162, 127, 164, 114, 89, 49, 163, 129, 61, 118, 87, 126, 143, 165, 71, 22, 23, 97, 43, 24, 9, 105, 144, 138, 119, 96, 54, 161, 104, 62, 63, 46, 101, 147, 88, 26, 153, 120, 103, 44, 117, 57, 132, 10, 151, 81, 91, 159, 68, 95, 70, 17, 11, 31, 50, 47, 158, 82, 112, 83, 18, 45, 64, 66, 109, 124, 29, 125, 115, 32, 111, 36, 37, 19, 134, 34, 51, 128, 3, 35, 100, 98, 110, 67, 136, 4, 137, 93, 122, 135, 92, 8, 6, 130, 52, 140, 5, 42, 142, 155, 20, 141, 113, 27, 131, 139, 154, 28, 123, 170, 2, 30, 90, 169, 33, 94, 7]
selected = [31, 78, 79, 67, 56, 8, 62, 42, 49, 18, 59, 44, 20, 80, 58, 73, 33, 2, 68, 37, 51, 34, 70, 40, 50, 63, 77, 64, 57, 53, 52, 48, 5, 72, 16, 45, 9, 35, 25, 39, 1, 19, 61, 38, 21, 55, 47, 11, 3, 32, 7, 13, 4, 65, 15, 46, 66, 74, 54, 36, 22, 14, 24, 43, 71, 69, 17, 75, 41, 12, 10, 6, 26, 27, 30, 23, 29, 28, 60, 76]

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
    kfold = KFold(n_splits=k_fold, random_state=0, shuffle=True)
    for train, test in kfold.split(X):
        X_train, X_test, y_train, y_test = prepare_data(train, test, X, y, num_input=X.shape[1]-1)

        if is_classification:
            if is_binary_classification :
                model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
            else:
                model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
            y_predict = model.fit(X_train, y_train).predict(X_test)
            auc += multiclass_roc_auc_score(y_test, y_predict, average="weighted")
            acc += accuracy_score(y_test, y_predict)
        else:
            model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
            y_predict = model.fit(X_train, y_train).predict(X_test)
            rmse += math.sqrt(mean_squared_error(y_test,y_predict))

    if is_classification:
        exp_output.insert(0,[i, acc / k_fold, auc/k_fold])
    else:
        exp_output.insert(0,[i, rmse / k_fold])

    del selected[-1]

with open("experiment_results_xg.csv", mode='w',newline='') as exp_file:
    if is_classification:
        exp_output.insert(0, ['K', 'acc', 'auc'])
    else:
        exp_output.insert(0, ['K', 'rmse'])
    wr = csv.writer(exp_file, dialect='excel')
    wr.writerows(exp_output)
