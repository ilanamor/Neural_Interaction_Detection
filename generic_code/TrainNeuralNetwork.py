import numpy as np
import math
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

#######################
# Train Network
#######################

#constant
l1_const = 5e-5
n_hidden_uni = 10 #for main effects network
num_output = 1 # regression or classification output dimension

#grt from user
use_main_effect_nets = None
learning_rate = None
num_epochs = None
batch_size = None
num_samples = None
num_hidden_layers = None
hidden_layers = None
num_input = None
df = None
tg_index = None
weights = {}
biases = {}

def set_parameters(_use_main_effect_nets,_learning_rate,_num_epochs,_batch_size,
                  _num_samples,_num_hidden_layers,_hidden_layers,_num_input,_df,_tg_index):
    global use_main_effect_nets, learning_rate,num_epochs,batch_size,num_samples,num_hidden_layers,hidden_layers,num_input,df,tg_index
    use_main_effect_nets = _use_main_effect_nets
    learning_rate = _learning_rate
    num_epochs = _num_epochs
    batch_size = _batch_size
    num_samples = _num_samples
    num_hidden_layers = _num_hidden_layers
    hidden_layers = _hidden_layers
    num_input = _num_input
    df = _df
    tg_index = _tg_index


def prepare_network():
    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_output])
    tf.set_random_seed(0)
    np.random.seed(0)
    # Get data
    tr_x, va_x, te_x, tr_y, va_y, te_y = prepare_data()
    tr_size = tr_x.shape[0]
    create_weights()
    create_biases()
    sess = construct_model(X, Y, tr_x, va_x, te_x, tr_y, va_y, te_y, tr_size)
    return sess,weights


def prepare_data():
    #Y = np.reshape(df.iloc[:,tg_index].values,(-1,1))
    sync_random = np.random.uniform(low=-1, high=1, size=(num_samples, 10))

    # Y = np.expand_dims(df.iloc[:, tg_index].values, axis=1)
    # dt_without_tg = df.drop(df.columns[[tg_index]], 1)
    # X = dt_without_tg.values

    X = df.drop(df.columns[[tg_index]], axis=1).values
    Y = np.expand_dims(df.iloc[:, tg_index].values, axis=1)

    a = num_samples // 3
    b = 2 * num_samples // 3

    tr_x, va_x, te_x = X[:a], X[a:b], X[b:]
    tr_y, va_y, te_y = Y[:a], Y[a:b], Y[b:]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(tr_x)
    scaler_y.fit(tr_y)

    tr_x, va_x, te_x = scaler_x.transform(tr_x), scaler_x.transform(va_x), scaler_x.transform(te_x)
    tr_y, va_y, te_y = scaler_y.transform(tr_y), scaler_y.transform(va_y), scaler_y.transform(te_y)
    return tr_x, va_x, te_x, tr_y, va_y, te_y


# access weights & biases
def create_weights():
    global weights
    weights['h1'] = tf.Variable(tf.truncated_normal([num_input, hidden_layers[0]], 0, 0.1))
    for x in range(1, num_hidden_layers):
        label = str(x+1)
        weights['h'+label] = tf.Variable(tf.truncated_normal([hidden_layers[x-1], hidden_layers[x]], 0, 0.1))
    weights['out'] = tf.Variable(tf.truncated_normal([hidden_layers[num_hidden_layers-1], num_output], 0, 0.1))


def create_biases():
    global biases
    for x in range(1, num_hidden_layers):
        label = str(x)
        biases['b'+label] = tf.Variable(tf.truncated_normal([hidden_layers[x-1]], 0, 0.1))
    biases['out'] = tf.Variable(tf.truncated_normal([num_output], 0, 0.1))


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
def construct_model(X,Y,tr_x, va_x, te_x, tr_y, va_y, te_y, tr_size):
    global weights
    global biases
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

    return sess
