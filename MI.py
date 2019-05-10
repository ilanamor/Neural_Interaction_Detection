import sklearn.feature_selection as sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# params ****************
file_path = r'C:\Users\karin\PycharmProjects\current\datasets\cal housing\cal_housing.csv'
header=True
index=False
is_classification = False
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
if (is_classification) :
    mi = sklearn.mutual_info_classif(X, Y, 'auto', 3, True, None)
else:
    mi=sklearn.mutual_info_regression(X, Y, 'auto', 3, True, None)
print(mi)

