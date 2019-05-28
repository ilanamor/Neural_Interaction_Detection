import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sklearn.feature_selection as sklearn

def read_csv():
    df = pd.read_csv(file_path) if header else pd.read_csv(file_path, header=None)
    df = df.drop(df.columns[[0]], axis=1) if index else df  # without the index column
    df.fillna(df.mean(), inplace=True)

    range = df.shape[1] - 1 if is_classification else df.shape[1]
    for y in df.columns[0:range]:
        if (df[y].dtype == np.float32):
            df[y] = df[y].astype('float64')
        elif (df[y].dtype == np.object):
            label_encoder = LabelEncoder()
            df[y] = label_encoder.fit_transform(df[y]).astype('float64')
        else:
            continue
    return df

def prepare_data(train, test, X_full, Y_full, num_input):
    tr_x, te_x, tr_y, te_y = X_full[train], X_full[test], Y_full[train], Y_full[test]

    for i in range(num_input):
        if df[df.columns[i]].dtype == np.float64:
            scaler_x = StandardScaler()
            scaler_x.fit(tr_x.T[i].reshape(-1, 1))
            tr_x_tmp, te_x_tmp = scaler_x.transform(tr_x.T[i].reshape(-1, 1)), scaler_x.transform(te_x.T[i].reshape(-1, 1))
            tr_x.T[i]=tr_x_tmp.flatten()
            te_x.T[i]=te_x_tmp.flatten()

    if not is_classification:
        scaler_y = StandardScaler()
        scaler_y.fit(tr_y)
        tr_y, te_y = scaler_y.transform(tr_y), scaler_y.transform(te_y)

    return tr_x, te_x, tr_y, te_y

# ******************************* PARAMS ***************************
file_path = r'C:\Users\karin\PycharmProjects\current\datasets\aps_failure\aps_failure.csv'
header = True
index = False
is_classification = True
# ****************************************************************

np.random.seed(0)
df = read_csv()
X = df.values[:,:-1,]
y = df.values[:,-1]
exp_output = []
if (is_classification) :
    mi = sklearn.mutual_info_classif(X, y, 'auto', 3, True, 123)
    print(len(mi))
    for i in range(len(mi)):
     exp_output.insert(0, [i+1, mi[i]])
else:
    mi = sklearn.mutual_info_regression(X, y, 'auto', 3, True, 123)


with open("experiment_results_mi.csv", mode='w',newline='') as exp_file:
    exp_output.insert(0, ['F-number', 'MI-score'])
    wr = csv.writer(exp_file, dialect='excel')
    wr.writerows(exp_output)