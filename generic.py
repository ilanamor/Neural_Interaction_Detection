import numpy as np
import math
import bisect
import operator
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#######################
# Train Neural Network
#######################

file_name = ""
header = None
use_main_effect_nets = True # toggle this to use "main effect" nets

# Constant Parameters
learning_rate = 0.01
num_epochs = 200
batch_size = 100
#mlp
l1_const = 5e-5
#mlp-cutoff
# l1_const = 1e-4
# Network Parameters
n_hidden_1 = 140 # 1st layer number of neurons #gui
n_hidden_2 = 100 # 2nd layer number of neurons
n_hidden_3 = 60 # 3rd "
n_hidden_4 = 20 # 4th "
n_hidden_uni = 10
num_hidden_layers=4
hidden_layers=[140,100,60,20]

num_output = 1 # regression or classification output dimension
out_name="out_letters_generic.png"
df=''
num_samples = 20000
num_input = 16 # simple synthetic example input dimension #num of features


# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_output])

tf.set_random_seed(0)
np.random.seed(0)

def read_csv():
    df = pd.read_csv(file_name, header=header)
    num_samples = 20000
    num_input = 16  # simple synthetic example input dimension #num of features
def gen_data():

    label_encoder = LabelEncoder()

    X = df.iloc[:, 1:17].values
    Y = np.expand_dims(label_encoder.fit_transform(df.iloc[:,0].values),axis=1)

    # tr_x, te_x, tr_y, te_y = train_test_split(X, Y, test_size=0.1, random_state=1)
    # tr_x, va_x, tr_y, va_y = train_test_split(tr_x, tr_y, test_size=0.1111111111, random_state=1)

    tr_x, te_x, tr_y, te_y = train_test_split(X, Y, test_size=0.2, random_state=1)
    te_x, va_x, te_y, va_y = train_test_split(te_x, te_y, test_size=0.5, random_state=1)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(tr_x)
    scaler_y.fit(tr_y)

    tr_x, va_x, te_x = scaler_x.transform(tr_x), scaler_x.transform(va_x), scaler_x.transform(te_x)
    tr_y, va_y, te_y = scaler_y.transform(tr_y), scaler_y.transform(va_y), scaler_y.transform(te_y)
    return tr_x, va_x, te_x, tr_y, va_y, te_y


# Get data
tr_x, va_x, te_x, tr_y, va_y, te_y = gen_data()
tr_size = tr_x.shape[0]

# access weights & biases
# weights1 = {
#     'h1': tf.Variable(tf.truncated_normal([num_input, n_hidden_1], 0, 0.1)),
#     'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
#     'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
#     'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
#     'out': tf.Variable(tf.truncated_normal([n_hidden_4, num_output], 0, 0.1))
# }

weights = {}

def create_weights():
    global weights
    weights['h1'] = tf.Variable(tf.truncated_normal([num_input, hidden_layers[0]], 0, 0.1))
    for x in range(1, num_hidden_layers):
        label = str(x+1)
        weights['h'+label] = tf.Variable(tf.truncated_normal([hidden_layers[x-1], hidden_layers[x]], 0, 0.1))
    weights['out'] = tf.Variable(tf.truncated_normal([hidden_layers[num_hidden_layers-1], num_output], 0, 0.1))

create_weights()

# biases1 = {
#     'b1': tf.Variable(tf.truncated_normal([n_hidden_1], 0, 0.1)),
#     'b2': tf.Variable(tf.truncated_normal([n_hidden_2], 0, 0.1)),
#     'b3': tf.Variable(tf.truncated_normal([n_hidden_3], 0, 0.1)),
#     'b4': tf.Variable(tf.truncated_normal([n_hidden_4], 0, 0.1)),
#     'out': tf.Variable(tf.truncated_normal([num_output], 0, 0.1))
# }

biases = {}
def create_biases():
    global biases
    for x in range(1, num_hidden_layers+1):
        label = str(x)
        biases['b'+label] = tf.Variable(tf.truncated_normal([hidden_layers[x-1]], 0, 0.1))
    biases['out'] = tf.Variable(tf.truncated_normal([num_output], 0, 0.1))

create_biases()

def get_weights_uninet():
    weights = {
        'h1': tf.Variable(tf.truncated_normal([1, n_hidden_uni], 0, 0.1)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_uni, n_hidden_uni], 0, 0.1)),
        'h3': tf.Variable(tf.truncated_normal([n_hidden_uni, n_hidden_uni], 0, 0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_uni, num_output], 0, 0.1))
    }
    return weights

def get_biases_uninet():
    biases = {
        'b1': tf.Variable(tf.truncated_normal([n_hidden_uni], 0, 0.1)),
        'b2': tf.Variable(tf.truncated_normal([n_hidden_uni], 0, 0.1)),
        'b3': tf.Variable(tf.truncated_normal([n_hidden_uni], 0, 0.1))
    }
    return biases

# Create model
def normal_neural_net(x, weights, biases):
    layer = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    for x in range(1, num_hidden_layers):
        label = str(x+1)
        layer_tmp = layer
        layer = tf.nn.relu(tf.add(tf.matmul(layer_tmp, weights['h'+label]), biases['b'+label]))
    out_layer = tf.matmul(layer, weights['out']) + biases['out']
    return out_layer

def main_effect_net(x, weights, biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    out_layer = tf.matmul(layer_3, weights['out'])
    return out_layer

# L1 regularizer
def l1_norm(a): return tf.reduce_sum(tf.abs(a))

# Construct model
net = normal_neural_net(X, weights, biases)

if use_main_effect_nets:
    me_nets = []
    for x_i in range(num_input):
        me_net = main_effect_net(tf.expand_dims(X[:,x_i],1), get_weights_uninet(), get_biases_uninet())
        me_nets.append(me_net)
    net = net + sum(me_nets)

# Define optimizer
loss_op = tf.losses.mean_squared_error(labels=Y, predictions=net)
# loss_op = tf.sigmoid_cross_entropy_with_logits(labels=Y,logits=net) # use this in the case of binary classification
sum_l1 = tf.reduce_sum([l1_norm(weights[k]) for k in weights])
loss_w_reg_op = loss_op + l1_const*sum_l1

batch = tf.Variable(0)
decaying_learning_rate = tf.train.exponential_decay(learning_rate, batch*batch_size, tr_size, 0.95, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=decaying_learning_rate).minimize(loss_w_reg_op, global_step=batch)

init = tf.global_variables_initializer()
n_batches = tr_size // batch_size
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

print('Initialized')

for epoch in range(num_epochs):

    batch_order = list(range(n_batches))
    np.random.shuffle(batch_order)

    for i in batch_order:
        batch_x = tr_x[i * batch_size:(i + 1) * batch_size]
        batch_y = tr_y[i * batch_size:(i + 1) * batch_size]
        _, lr = sess.run([optimizer, decaying_learning_rate], feed_dict={X: batch_x, Y: batch_y})

    if (epoch + 1) % 50 == 0:
        tr_mse = sess.run(loss_op, feed_dict={X: tr_x, Y: tr_y})
        va_mse = sess.run(loss_op, feed_dict={X: va_x, Y: va_y})
        te_mse = sess.run(loss_op, feed_dict={X: te_x, Y: te_y})
        print('Epoch', epoch + 1)
        print('\t', 'train rmse', math.sqrt(tr_mse), 'val rmse', math.sqrt(va_mse), 'test rmse', math.sqrt(te_mse))
        print('\t', 'learning rate', lr)

print('done')

###################
#Interpret Weights
###################

def preprocess_weights(w_dict):
    hidden_layers = [int(layer[1:]) for layer in w_dict.keys() if layer.startswith('h')]
    output_h = ['h' + str(x) for x in range(max(hidden_layers),1,-1)]
    w_agg = np.abs(w_dict['out'])
    w_h1 = np.abs(w_dict['h1'])

    for h in output_h:
        w_agg = np.matmul( np.abs(w_dict[h]), w_agg)

    return w_h1, w_agg


def get_interaction_ranking(w_dict):
    xdim = w_dict['h1'].shape[0]
    w_h1, w_agg = preprocess_weights(w_dict)

    # rank interactions
    interaction_strengths = dict()

    for i in range(len(w_agg)):
        sorted_fweights = sorted(enumerate(w_h1[:, i]), key=lambda x: x[1], reverse=True)
        interaction_candidate = []
        weight_list = []
        for j in range(len(w_h1)):
            bisect.insort(interaction_candidate, sorted_fweights[j][0] + 1)
            weight_list.append(sorted_fweights[j][1])
            if len(interaction_candidate) == 1:
                continue
            interaction_tup = tuple(interaction_candidate)
            if interaction_tup not in interaction_strengths:
                interaction_strengths[interaction_tup] = 0
            inter_agg = min(weight_list)
            interaction_strengths[interaction_tup] += np.abs(inter_agg * np.sum(w_agg[i]))

    interaction_sorted = sorted(interaction_strengths.items(), key=operator.itemgetter(1), reverse=True)
    # forward prune the ranking of redundant interactions
    interaction_ranking_pruned = []
    existing_largest = []
    for i, inter in enumerate(interaction_sorted):
        if len(interaction_ranking_pruned) > 20000: break
        skip = False
        indices_to_remove = set()
        for inter2_i, inter2 in enumerate(existing_largest):
            # if this is not the existing largest
            if set(inter[0]) < set(inter2[0]):
                skip = True
                break
            # if this is larger, then need to recall this index later to remove it from existing_largest
            if set(inter[0]) > set(inter2[0]):
                indices_to_remove.add(inter2_i)
        if skip:
            assert len(indices_to_remove) == 0
            continue
        prevlen = len(existing_largest)
        existing_largest[:] = [el for el_i, el in enumerate(existing_largest) if el_i not in indices_to_remove]
        existing_largest.append(inter)
        interaction_ranking_pruned.append((inter[0], inter[1]))

        curlen = len(existing_largest)

    return interaction_ranking_pruned

def get_pairwise_ranking(w_dict):
    xdim = w_dict['h1'].shape[0]
    w_h1, w_agg = preprocess_weights(w_dict)

    input_range = range(1,xdim+1)
    pairs = [(xa,yb) for xa in input_range for yb in input_range if xa != yb]
    for entry in pairs:
        if (entry[1], entry[0]) in pairs:
            pairs.remove((entry[1],entry[0]))

    pairwise_strengths = []
    heatmap_df= [[0]*num_input for i in range(num_input)]
    for pair in pairs:
        a = pair[0]
        b = pair[1]
        wa = w_h1[a-1].reshape(w_h1[a-1].shape[0],1)
        wb = w_h1[b-1].reshape(w_h1[b-1].shape[0],1)
        wz = np.abs(np.minimum(wa , wb))*w_agg
        cab = np.sum(np.abs(wz))
        pairwise_strengths.append((pair, cab))
        #save in a list
        heatmap_df[b-1][a-1] = cab

    #create heat-map
    sns.set()
    heatmap_df = pd.DataFrame(np.array(heatmap_df))
    heatmap_df.index += 1
    heatmap_df.columns += 1
    ax = sns.heatmap(heatmap_df, cmap='Blues')
    plt.savefig("heat_maps\\"+out_name)

    #soet pairwise list
    pairwise_ranking = sorted(pairwise_strengths,key=operator.itemgetter(1), reverse=True)
    return pairwise_ranking

w_dict = sess.run(weights)

# Variable-Order Interaction Ranking
print(get_interaction_ranking(w_dict))

# Pairwise Interaction Ranking
print(get_pairwise_ranking(w_dict))

