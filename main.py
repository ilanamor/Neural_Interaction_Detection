import numpy as np
import math
import bisect
import operator
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

#######################
# get params
#######################

use_main_effect_nets = True # toggle this to use "main effect" nets #gui

# Parameters
learning_rate = 0.01 #gui
num_epochs = 200 #gui
batch_size = 100 #gui

num_samples = 30000 #30k datapoints, split 1/3-1/3-1/3

# Network Parameters
number_of_hide_unit =0 #gui
units_arr =[] #gui
class_name ="" #gui
num_input = 10 #num of features



