from scipy import stats
import pandas as pd
# ****params***
# file contains the difference d of MI and NID
file_path = r'C:\Users\karin\PycharmProjects\current\experiments\differanceWilx.csv'
# *********************************************************

X_diff=[]
df = pd.read_csv(file_path, header=None)
all = df.values[:,:,]
for i in range(all.shape[0]):
    X_diff.append(all[i][0])
print(stats.wilcoxon(X_diff))


