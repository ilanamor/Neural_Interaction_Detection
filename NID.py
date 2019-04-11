import csv
import os
import numpy as np
import math
import bisect
import operator
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class NID:
    # Just disables the warning, doesn't enable AVX/FMA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    learning_rate = 0.01
    num_epochs = 200
    batch_size = 100
    l1_const = 5e-5
    l2_const = 1e-4
    n_hidden_uni = 10
    global_pairwise_strengths = {}
    global_interaction_strengths = {}

    # Random seeds
    tf.set_random_seed(0)
    np.random.seed(0)

    def __init__(self, main_effects, is_index, is_header, file_path, output_path, hidden_layers_structure, is_classification_data):
        self.use_main_effect_nets = True if main_effects == 1 else False
        self.header = True if is_header == 1 else False
        self.index = True if is_index == 1 else False
        self.file_name = file_path
        self.out_path = output_path
        self.hidden_layers = hidden_layers_structure
        self.n_hidden_layers = len(hidden_layers_structure)
        #output
        self.heatmap_name=self.out_path+"\pairwise_heatmap.png"
        self.pairwise_out_name = self.out_path + "\pairwise_ranking.csv"
        self.higher_order_out_name = self.out_path + "\higher_order_ranking.csv"
        # set params
        self.is_classification = is_classification_data
        self.df, self.num_samples, self.num_input, self.num_output = self.read_csv()
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_output])
        # access weights & biases
        self.weights = self.create_weights()
        self.biases = self.create_biases()

    def read_csv(self):
        df = pd.read_csv(self.file_name) if self.header else pd.read_csv(self.file_name, header=None)
        df = df.drop(df.columns[[0]], axis=1) if self.index else df #without the index column
        df = self.preprocess_df(df)
        num_samples = df.shape[0]
        num_input = df.shape[1]-1 #without the target column
        num_out = df[df.columns[-1]].nunique() if self.is_classification else 1
        return df,num_samples,num_input,num_out

    def preprocess_df(self, df):
        range = df.shape[1]-1 if self.is_classification else df.shape[1]
        for y in df.columns[0:range]:
            if (df[y].dtype == np.int32 or df[y].dtype == np.int64 or df[y].dtype == np.float32):
                df[y] = df[y].astype('float64')
            elif (df[y].dtype == np.object):
                label_encoder = LabelEncoder()
                df[y] = label_encoder.fit_transform(df[y]).astype('float64')
            else:
                continue
        return df

    # Main function - running flow
    def run(self):
        X_full,Y_full = self.prepare_df()
        kfold = KFold(n_splits=5, random_state=None, shuffle=False)
        for train, test in kfold.split(X_full):
            tr_x, te_x, tr_y, te_y, va_x, va_y = self.prepare_data(train,test,X_full,Y_full)
            tr_size = tr_x.shape[0]
            sess = self.construct_model(tr_x, te_x, tr_y, te_y, va_x, va_y, tr_size)
            self.interpret_weights(sess)
            break
        self.average_results()
        self.create_heat_map()

    # Prepare the df - create X,Y
    def prepare_df(self):
        label_encoder = LabelEncoder()
        X_data = self.df.iloc[:, 0:self.num_input].values
        Y_data = pd.get_dummies(label_encoder.fit_transform(self.df.iloc[:,-1]),dtype=int).values if self.is_classification else np.expand_dims(self.df.iloc[:, -1].values, axis=1)
        return X_data,Y_data

    # Train & Test split
    def prepare_data(self, train, test, X_full, Y_full):
        tr_x, te_x, tr_y, te_y = X_full[train], X_full[test], Y_full[train], Y_full[test]
        te_x, va_x, te_y, va_y = train_test_split(te_x, te_y, test_size = 0.5)
        # tr_x, va_x, tr_y, va_y = train_test_split(tr_x, tr_y, test_size=0.5)
        scaler_x = StandardScaler()
        scaler_x.fit(tr_x)
        tr_x, te_x, va_x = scaler_x.transform(tr_x), scaler_x.transform(te_x), scaler_x.transform(va_x)

        if not self.is_classification:
            scaler_y = StandardScaler()
            scaler_y.fit(tr_y)
            tr_y, te_y, va_y = scaler_y.transform(tr_y), scaler_y.transform(te_y), scaler_y.transform(va_y)

        return tr_x, te_x, tr_y, te_y, va_x, va_y

    # Network weights
    def create_weights(self):
        weights = {}
        weights['h1'] = tf.Variable(tf.truncated_normal([self.num_input, self.hidden_layers[0]], 0, 0.1))
        for x in range(1, self.n_hidden_layers):
            label = str(x+1)
            weights['h'+label] = tf.Variable(tf.truncated_normal([self.hidden_layers[x-1], self.hidden_layers[x]], 0, 0.1))
        weights['out'] = tf.Variable(tf.truncated_normal([self.hidden_layers[self.n_hidden_layers-1], self.num_output], 0, 0.1))
        return weights

    # Network biases
    def create_biases(self):
        biases = {}
        for x in range(1, self.n_hidden_layers+1):
            label = str(x)
            biases['b'+label] = tf.Variable(tf.truncated_normal([self.hidden_layers[x-1]], 0, 0.1))
        biases['out'] = tf.Variable(tf.truncated_normal([self.num_output], 0, 0.1))
        return biases

    # Uninets for main effects - weights
    def get_weights_uninet(self):
        weights = {
            'h1': tf.Variable(tf.truncated_normal([1, self.n_hidden_uni], 0, 0.1)),
            'h2': tf.Variable(tf.truncated_normal([self.n_hidden_uni, self.n_hidden_uni], 0, 0.1)),
            'h3': tf.Variable(tf.truncated_normal([self.n_hidden_uni, self.n_hidden_uni], 0, 0.1)),
            'out': tf.Variable(tf.truncated_normal([self.n_hidden_uni, self.num_output], 0, 0.1))
        }
        return weights

    # Uninets for main effects - biases
    def get_biases_uninet(self):
        biases = {
            'b1': tf.Variable(tf.truncated_normal([self.n_hidden_uni], 0, 0.1)),
            'b2': tf.Variable(tf.truncated_normal([self.n_hidden_uni], 0, 0.1)),
            'b3': tf.Variable(tf.truncated_normal([self.n_hidden_uni], 0, 0.1))
        }
        return biases

    # Create model
    def normal_neural_net(self, x, weights, biases):
        layer = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        for i in range(2, self.n_hidden_layers+1):
            layer_tmp = layer
            layer = tf.nn.relu(tf.add(tf.matmul(layer_tmp, weights['h' + str(i)]), biases['b' + str(i)]))
        out_layer = tf.matmul(layer, weights['out']) + biases['out']
        return out_layer


    def main_effect_net(self, x, weights, biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
        out_layer = tf.matmul(layer_3, weights['out'])
        return out_layer

    def individual_univariate_net(self, x, weights, biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
        out_layer = tf.matmul(layer_3, weights['out'])
        return out_layer

    # L1 regularizer
    def l1_norm(self, a): return tf.reduce_sum(tf.abs(a))

    # L2 regularizer
    def l2_norm(self, a):
        return tf.reduce_sum(tf.pow(a, 2))

    # Construct the model
    def construct_model(self, tr_x, te_x, tr_y, te_y, va_x, va_y, tr_size):
        # Construct model
        net = self.normal_neural_net(self.X, self.weights, self.biases)
        # check main effects need
        if self.use_main_effect_nets:
            me_nets = []
            for x_i in range(self.num_input):
                me_net = self.main_effect_net(tf.expand_dims(self.X[:, x_i], 1), self.get_weights_uninet(),
                                              self.get_biases_uninet())
                me_nets.append(me_net)
            net = net + sum(me_nets)

        # Define optimizer
        # loss_op = (tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=net) if self.num_output == 2 else tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y,logits=net)) if self.is_classification else tf.losses.mean_squared_error(labels=self.Y, predictions=net)

        if self.is_classification:
            if self.num_output == 2:
                loss_op = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=net) # use this in the case of binary classification
                loss_op = tf.reduce_mean(loss_op)
            else:
                loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y,logits=net)
                loss_op = tf.reduce_mean(loss_op)
        else:
           loss_op = tf.losses.mean_squared_error(labels=self.Y, predictions=net)

        sum_l1 = tf.reduce_sum([self.l1_norm(self.weights[k]) for k in self.weights])
        loss_w_reg_op = loss_op + self.l1_const * sum_l1

        batch = tf.Variable(0)
        decaying_learning_rate = tf.train.exponential_decay(self.learning_rate, batch * self.batch_size, tr_size, 0.95,
                                                            staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=decaying_learning_rate).minimize(loss_w_reg_op,
                                                                                          global_step=batch)

        # init = tf.global_variables_initializer()
        n_batches = tr_size // self.batch_size
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.25
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # sess.run(init)
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        print('Initialized')

        for epoch in range(self.num_epochs):

            batch_order = list(range(n_batches))
            np.random.shuffle(batch_order)

            for i in batch_order:
                batch_x = tr_x[i * self.batch_size:(i + 1) * self.batch_size]
                batch_y = tr_y[i * self.batch_size:(i + 1) * self.batch_size]
                _, lr = sess.run([optimizer, decaying_learning_rate], feed_dict={self.X: batch_x, self.Y: batch_y})

            if (epoch + 1) % 50 == 0:
                if self.is_classification:
                    print('Epoch', epoch + 1)
                    # Test model
                    pred = tf.nn.sigmoid(net) if self.num_output == 2 else tf.nn.softmax(net)# Apply softmax to logits
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print('\t', 'train acc', accuracy.eval(feed_dict={self.X: tr_x, self.Y: tr_y},session=sess), 'val acc', accuracy.eval(feed_dict={self.X: va_x, self.Y: va_y},session=sess), 'test acc', accuracy.eval(feed_dict={self.X: te_x, self.Y: te_y},session=sess))

                    #auc
                    auc, auc_op = tf.metrics.auc(labels=self.Y, predictions=pred)
                    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(self.Y,1), predictions=tf.argmax(pred,1))
                    sess.run(tf.local_variables_initializer())

                    v = sess.run([auc, auc_op], feed_dict={self.X: tr_x,
                                                           self.Y: tr_y})
                    print('auc tr:', v)

                    r = sess.run([acc, acc_op], feed_dict={self.X: tr_x,
                                                           self.Y: tr_y})
                    print('acc tr:', r)

                    v = sess.run([auc, auc_op], feed_dict={self.X: va_x,
                                                           self.Y: va_y})
                    print('auc va:',v)

                    r = sess.run([acc, acc_op], feed_dict={self.X: va_x,
                                                           self.Y: va_y})
                    print('acc va:', r)

                    v = sess.run([auc, auc_op], feed_dict={self.X: te_x,
                                                           self.Y: te_y})
                    print('auc te:', v)
                    print('\t', 'learning rate', lr)
                else:
                    tr_mse = sess.run(loss_op, feed_dict={self.X: tr_x, self.Y: tr_y})
                    va_mse = sess.run(loss_op, feed_dict={self.X: va_x, self.Y: va_y})
                    te_mse = sess.run(loss_op, feed_dict={self.X: te_x, self.Y: te_y})
                    print('Epoch', epoch + 1)
                    print('\t', 'train rmse', math.sqrt(tr_mse), 'val rmse', math.sqrt(va_mse), 'test rmse',
                          math.sqrt(te_mse))
                    print('\t', 'learning rate', lr)

        print('done')
        return sess


    def preprocess_weights(self, w_dict):
        hidden_layers = [int(layer[1:]) for layer in w_dict.keys() if layer.startswith('h')]
        output_h = ['h' + str(x) for x in range(max(hidden_layers),1,-1)]
        w_agg = np.abs(w_dict['out'])
        w_h1 = np.abs(w_dict['h1'])

        for h in output_h:
            w_agg = np.matmul( np.abs(w_dict[h]), w_agg)

        return w_h1, w_agg

    # High-order interactions
    def get_interaction_ranking(self, w_dict):
        xdim = w_dict['h1'].shape[0]
        w_h1, w_agg = self.preprocess_weights(w_dict)

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
            if len(interaction_ranking_pruned) > 20000 : break
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
            if (inter[0] in self.global_interaction_strengths.keys()):
                self.global_interaction_strengths[inter[0]] = (inter[1] + self.global_interaction_strengths[inter[0]])
            else:
                self.global_interaction_strengths[inter[0]] = inter[1]
            curlen = len(existing_largest)

        return interaction_ranking_pruned

    # Pairwise interactions
    def get_pairwise_ranking(self, w_dict):
        xdim = w_dict['h1'].shape[0]
        w_h1, w_agg = self.preprocess_weights(w_dict)

        input_range = range(1,xdim+1)
        pairs = [(xa,yb) for xa in input_range for yb in input_range if xa != yb]
        for entry in pairs:
            if (entry[1], entry[0]) in pairs:
                pairs.remove((entry[1],entry[0]))

        pairwise_strengths_round = []
        for pair in pairs:
            a = pair[0]
            b = pair[1]
            wa = w_h1[a-1].reshape(w_h1[a-1].shape[0],1)
            wb = w_h1[b-1].reshape(w_h1[b-1].shape[0],1)
            wz = np.abs(np.minimum(wa , wb))*w_agg
            cab = np.sum(np.abs(wz))
            pairwise_strengths_round.append((pair, cab))
            if (pair in self.global_pairwise_strengths.keys()):
                self.global_pairwise_strengths[pair] = (cab + self.global_pairwise_strengths[pair])
            else:
                self.global_pairwise_strengths[pair] = cab

        #sort pairwise list
        pairwise_ranking = sorted(pairwise_strengths_round,key=operator.itemgetter(1), reverse=True)
        return pairwise_ranking

    # create HetMap
    def create_heat_map(self):
        pairwise_2d = [[0] * self.num_input for i in range(self.num_input)]
        for pair,cab in self.global_pairwise_strengths.items():
            a = pair[0]
            b = pair[1]
            pairwise_2d[b - 1][a - 1] = cab
            # pairwise_2d[a - 1][b - 1] = cab
        # sns.set()
        heatmap_df = pd.DataFrame(np.array(pairwise_2d))
        heatmap_df.index += 1
        heatmap_df.columns += 1
        #print heatmap
        sns.set()
        ax = sns.heatmap(heatmap_df, cmap='Blues',cbar_kws={"shrink": .5})
        if os.path.isfile(self.heatmap_name):
            os.remove(self.heatmap_name)
        plt.savefig(self.heatmap_name)
        # plt.show()


    def interpret_weights(self, sess):
        w_dict = sess.run(self.weights)

        # Higher-Order Interaction Ranking
        print('\nHigher-Order Interaction Ranking')
        print(self.get_interaction_ranking(w_dict))

        # Pairwise Interaction Ranking
        print('\nPairwise Interaction Ranking')
        print(self.get_pairwise_ranking(w_dict))

    # final results
    def average_results(self):
        for pair, cab in self.global_pairwise_strengths.items():
            self.global_pairwise_strengths[pair] = float(cab) / 5

        pairwise_ranking = sorted(self.global_pairwise_strengths.items(), key=operator.itemgetter(1), reverse=True)

        for interaction, cab in self.global_interaction_strengths.items():
            self.global_interaction_strengths[interaction] = float(cab) / 5

        interaction_ranking = sorted(self.global_interaction_strengths.items(), key=operator.itemgetter(1), reverse=True)

        print('\nFinal results:','\n##############')
        print('\nHigher-Order Interaction Ranking')
        print(interaction_ranking)
        print('\nPairwise Interaction Ranking')
        print(pairwise_ranking)
        self.write_to_csv(interaction_ranking, self.higher_order_out_name)
        self.write_to_csv(pairwise_ranking, self.pairwise_out_name)


    def write_to_csv(self, interactions, name):
        with open(name , 'w') as out:
            csv_out = csv.writer(out, lineterminator='\n')
            csv_out.writerow(['Features', 'Strength'])
            for row in interactions:
                csv_out.writerow(row)


