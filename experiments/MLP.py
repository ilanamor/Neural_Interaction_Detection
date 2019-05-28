import csv
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
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

def prepare_data(train, test, X_full, Y_full):
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

file_path = r'C:\Users\Ilana\PycharmProjects\Neural_Interaction_Detection\datasets\ozone\eighthr.csv'
header = False
index = False
is_classification = True
k_fold = 10
# selected = [13,15,8,9,5,10,3,6,11,12,16,1,4,7,14,2]
# selected = [14,11,7,12,10,6,9,16,13,3,8,1,2,15,5,4]
# selected = [5,7,2,6,4,9,10,8,11,1,12,3]
# selected = [4,9,10,11,3,2,1,7,12,6,8,5]
# selected = [1,2,5,6,4,3,8,7]
# selected = [1,8,2,4,3,7,5,6]
# selected = [3,8,20,12,22,23,1,14,25,26,2,9,30,17,24,6,11,10,4,15,18,27,7,29,13,5,19,28,21,16]
# selected = [1,5,2,7,13,3,6,28,29,27,25,14,24,12,26,11,23,10,30,4,22,8,20,18,15,17,19,9,16,21]
# selected = [19, 148, 118, 54, 9, 7, 143, 114, 78, 57, 166, 17, 40, 146, 16, 66, 81, 126, 52, 55, 86, 13, 26, 139, 5, 122, 6, 106, 138, 20, 58, 142, 43, 141, 60, 48, 68, 115, 104, 70, 131, 27, 79, 63, 140, 93, 108, 164, 46, 128, 31, 83, 111, 94, 25, 155, 65, 145, 125, 101, 2, 49, 62, 22, 110, 113, 134, 36, 92, 112, 144, 162, 29, 95, 154, 119, 137, 77, 14, 98, 121, 73, 136, 124, 53, 97, 150, 87, 165, 45, 35, 84, 129, 159, 100, 103, 42, 130, 50, 120, 135, 153, 41, 32, 109, 149, 24, 76, 123, 34, 158, 64, 96, 61, 80, 3, 51, 15, 10, 59, 133, 147, 12, 152, 102, 30, 8, 91, 88, 89, 163, 71, 18, 82, 56, 105, 151, 69, 4, 99, 156, 67, 72, 38, 107, 74, 23, 1, 167, 21, 47, 160, 132, 37, 28, 90, 157, 75, 44, 11, 33, 127, 117, 161, 85, 39, 116]
# selected = [1, 152, 163, 93, 37, 167, 103, 164, 111, 4, 43, 110, 145, 69, 100, 80, 166, 96, 8, 127, 97, 161, 46, 154, 36, 132, 105, 157, 123, 15, 144, 112, 162, 67, 165, 155, 125, 72, 156, 73, 63, 133, 81, 99, 70, 90, 26, 33, 78, 30, 71, 101, 20, 79, 126, 75, 13, 106, 18, 24, 39, 5, 64, 136, 58, 124, 45, 66, 31, 149, 84, 120, 138, 113, 9, 119, 50, 104, 10, 60, 139, 150, 12, 114, 83, 76, 107, 59, 121, 98, 23, 115, 135, 56, 62, 16, 53, 86, 118, 85, 137, 2, 41, 88, 143, 40, 28, 27, 65, 21, 128, 87, 91, 34, 122, 102, 153, 49, 42, 74, 47, 131, 117, 160, 94, 116, 7, 140, 51, 151, 22, 109, 146, 14, 92, 3, 52, 48, 19, 54, 147, 82, 61, 38, 129, 29, 142, 25, 55, 57, 130, 68, 35, 108, 32, 95, 134, 159, 141, 44, 11, 89, 17, 6, 158, 77, 148]
# selected = [65, 77, 78, 60, 80, 72, 168, 39, 76, 121, 85, 102, 156, 1, 152, 79, 15, 53, 107, 74, 106, 75, 148, 145, 38, 56, 48, 133, 108, 16, 73, 150, 14, 25, 157, 59, 41, 40, 99, 69, 166, 116, 21, 84, 149, 55, 13, 12, 58, 146, 160, 167, 86, 162, 127, 164, 114, 89, 49, 163, 129, 61, 118, 87, 126, 143, 165, 71, 22, 23, 97, 43, 24, 9, 105, 144, 138, 119, 96, 54, 161, 104, 62, 63, 46, 101, 147, 88, 26, 153, 120, 103, 44, 117, 57, 132, 10, 151, 81, 91, 159, 68, 95, 70, 17, 11, 31, 50, 47, 158, 82, 112, 83, 18, 45, 64, 66, 109, 124, 29, 125, 115, 32, 111, 36, 37, 19, 134, 34, 51, 128, 3, 35, 100, 98, 110, 67, 136, 4, 137, 93, 122, 135, 92, 8, 6, 130, 52, 140, 5, 42, 142, 155, 20, 141, 113, 27, 131, 139, 154, 28, 123, 170, 2, 30, 90, 169, 33, 94, 7]
# selected = [78,1,77,79,76,75,74,73,72,18,22,8,64,21,9,100,65,133,3,145,156,10,27,137,149,134,66,111,87,135,86,138,144,92,139,136,142,141,98,31,20,115,140,101,102,67,91,150,153,32,44,43,151,148,113,114,108,45,93,143,147,146,158,62,152,155,154,4,169,5,170,69,81,85,128,61,110,124,6,71,68,17,132,42,131,99,23,125,2,24,130,97,109,95,127,26,15,83,112,121,82,126,63,25,129,94,56,50,161,119,57,55,80,107,165,70,167,162,157,46,116,163,58,13,89,104,117,60,160,40,14,164,11,53,106,54,103,122,41,105,84,59,166,37,51,48,118,159,88,19,39,12,47,120,29,7,49,16,38,96,52,33,168,90,36,35,30,34,123,28]
# selected = [78, 105, 25, 30, 89, 128, 137, 4, 58, 93, 90, 57, 141, 64, 154, 3, 5, 84, 104, 157, 21, 91, 139, 150, 11, 40, 7, 122, 31, 110, 81, 35, 2, 46, 51, 73, 52, 107, 140, 108, 22, 96, 114, 121, 112, 100, 94, 54, 55, 162, 146, 127, 123, 71, 99, 111, 163, 15, 149, 36, 72, 16, 143, 75, 61, 92, 117, 103, 133, 47, 109, 101, 106, 145, 44, 102, 39, 76, 136, 129, 12, 155, 28, 98, 134, 60, 79, 68, 135, 83, 158, 70, 14, 69, 161, 45, 74, 8, 9, 130, 116, 23, 147, 17, 56, 165, 159, 37, 1, 38, 138, 87, 119, 80, 152, 32, 20, 151, 115, 62, 88, 132, 156, 6, 148, 67, 97, 42, 124, 34, 24, 153, 26, 125, 82, 33, 43, 13, 118, 41, 85, 113, 65, 164, 50, 166, 48, 120, 77, 18, 29, 142, 86, 53, 49, 27, 66, 19, 95, 63, 59, 126, 144, 10, 131, 160]
# selected = [31, 78, 79, 67, 56, 8, 62, 42, 49, 18, 59, 44, 20, 80, 58, 73, 33, 2, 68, 37, 51, 34, 70, 40, 50, 63, 77, 64, 57, 53, 52, 48, 5, 72, 16, 45, 9, 35, 25, 39, 1, 19, 61, 38, 21, 55, 47, 11, 3, 32, 7, 13, 4, 65, 15, 46, 66, 74, 54, 36, 22, 14, 24, 43, 71, 69, 17, 75, 41, 12, 10, 6, 26, 27, 30, 23, 29, 28, 60, 76]
selected = [37, 55, 1, 22, 24, 56, 7, 59, 21, 64, 36, 19, 20, 70, 10, 11, 17, 61, 3, 71, 23, 6, 18, 38, 68, 58, 66, 63, 35, 45, 16, 69, 4, 57, 54, 41, 14, 31, 40, 65, 39, 12, 33, 32, 8, 13, 25, 50, 9, 53, 34, 2, 60, 62, 67, 30, 29, 5, 72, 28, 15, 51, 43, 49, 42, 27, 47, 44, 26, 46, 48, 52]

exp_output = []
# ****************************************************************

for i in range(len(selected),0,-1):
    np.random.seed(0)
    df = read_csv()
    X_all = df.iloc[:, 0:df.shape[1]-1].values
    y = df.values[:,-1]
    X = get_slice_of_data(selected, X_all)

    auc = 0
    acc = 0
    rmse = 0
    kfold = KFold(n_splits=k_fold, random_state=1992, shuffle=True)
    for train, test in kfold.split(X):
        X_train, X_test, y_train, y_test = prepare_data(train, test, X, y, num_input=X.shape[1]-1)

        if is_classification:
            model = MLPClassifier(random_state=123)
            y_predict = model.fit(X_train, y_train).predict(X_test)
            auc += multiclass_roc_auc_score(y_test, y_predict, average="weighted")
            acc += accuracy_score(y_test, y_predict)
        else:
            model = MLPRegressor(random_state=123)
            y_predict = model.fit(X_train, y_train).predict(X_test)
            rmse += math.sqrt(mean_squared_error(y_test,y_predict))

    if is_classification:
        exp_output.insert(0,[i, acc / k_fold, auc/k_fold])
    else:
        exp_output.insert(0,[i, rmse / k_fold])

    del selected[-1]

with open("experiment_results_mlp.csv", mode='w',newline='') as exp_file:
    if is_classification:
        exp_output.insert(0, ['K', 'acc', 'auc'])
    else:
        exp_output.insert(0, ['K', 'rmse'])
    wr = csv.writer(exp_file, dialect='excel')
    wr.writerows(exp_output)
