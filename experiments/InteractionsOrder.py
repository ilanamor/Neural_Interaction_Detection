import csv
from collections import OrderedDict

file_path = r'C:\Users\Ilana\PycharmProjects\Neural_Interaction_Detection\datasets\musk\tmp2.txt'
s = OrderedDict()
with open(file_path, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        for word in row:
            if word != '':
                s[int(word)] = 0

print(s.keys())