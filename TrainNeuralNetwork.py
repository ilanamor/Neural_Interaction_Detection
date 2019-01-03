import math
import bisect
import operator
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

#######################
# Train
#######################

l1_const = 5e-5
n_hidden_uni = 10
num_output = 1 # regression or classification output dimension

