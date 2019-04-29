import csv
import logging
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


class NID:
    # Just disables the warning, doesn't enable AVX/FMA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    learning_rate = 0.01
    l1_const = 5e-5
    l2_const = 1e-4
    n_hidden_uni = 10
    global_pairwise_strengths = {}
    global_interaction_strengths = {}

    # Random seeds
    tf.set_random_seed(0)
    np.random.seed(0)

    '''input params, init format:
    use_main_effect_nets - whether use main effects or not (true / false)
    use_cutoff - whether use cutoff or not (true / false)
    is_index_col - is index column exists (1 true / 0 false)
    is_header - is header exists (1 true / 0 false)
    file_path - full path
    out_path - full path
    units_list - network architecture (list of numbers seperate by comma)
    is_classification_col - is classification dataset (1 true / 0 false, false means regression)
    k_fold_entry - number of folds (int, greater than 2)
    num_epochs_entry -  number of epochs (int, greater than 1)
    batch_size_entry -  number of batches (int, greater than 1)
    '''

    def __init__(self, main_effects, cutoff, is_index, is_header, file_path, output_path, hidden_layers_structure, is_classification_data, k_fold_num = 5, num_of_epochs = 200, batch_size = 100):
        self.use_main_effect_nets = main_effects
        self.use_cutoff = cutoff
        self.header = True if is_header == 1 else False
        self.index = True if is_index == 1 else False
        self.file_name = file_path
        self.out_path = output_path
        self.hidden_layers = hidden_layers_structure
        self.n_hidden_layers = len(hidden_layers_structure)
        self.num_epochs = num_of_epochs
        self.batch_size = batch_size
        self.k_fold = k_fold_num

        #output
        self.heatmap_name=self.out_path+"/pairwise_heatmap.png"
        self.pairwise_out_name = self.out_path + "/pairwise_ranking.csv"
        self.higher_order_out_name = self.out_path + "/higher_order_ranking.csv"
        self.log_out_name = self.out_path + "/log.log"
        # set params
        self.is_classification = is_classification_data
        self.df, self.num_samples, self.num_input, self.num_output = self.read_csv()
        self.va_error = 0
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_output])

        # logger
        logging.basicConfig(filename=self.log_out_name,
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        logging.info("Running NID framework")

        self.logger = logging.getLogger('NID_LOG')

    def read_csv(self):
        df = pd.read_csv(self.file_name) if self.header else pd.read_csv(self.file_name, header=None)
        df = df.drop(df.columns[[0]], axis=1) if self.index else df #without the index column
        df = self.preprocess_df(df)
        num_samples = df.shape[0]
        num_input = df.shape[1]-1 #without the target column
        num_out = df[df.columns[-1]].nunique() if self.is_classification else 1
        if df.empty:
            raise ValueError('Empty file input')
        elif self.k_fold>num_samples:
            raise ValueError('Mismatch between K-folds and dataset')
        else:
            return df,num_samples,num_input,num_out


    def preprocess_df(self, df):

        # check incompatible dataset type and target type
        if self.incompatible_types(df[df.columns[-1]].dtype):
            raise ValueError('Incompatible dataset type and target type')

        range = df.shape[1] - 1 if self.is_classification else df.shape[1]
        for y in df.columns[0:range]:
            if (df[y].dtype == np.int32 or df[y].dtype == np.int64 or df[y].dtype == np.float32):
                df[y] = df[y].astype('float64')
            elif (df[y].dtype == np.object):
                label_encoder = LabelEncoder()
                df[y] = label_encoder.fit_transform(df[y]).astype('float64')
            else:
                continue
        return df

    # handle incompatible types
    def incompatible_types(self,target_type):
        if (self.is_classification and target_type in (np.float32, np.float64)):
            return True
        elif (not self.is_classification and target_type==np.object):
            return True
        return False

    # Main function - running flow
    def run(self):
        X_full,Y_full = self.prepare_df()
        kfold = KFold(n_splits=self.k_fold, random_state=None, shuffle=False)
        for train, test in kfold.split(X_full):
            # access weights & biases
            self.weights = self.create_weights()
            self.biases = self.create_biases()
            tr_x, te_x, tr_y, te_y, va_x, va_y = self.prepare_data(train,test,X_full,Y_full)
            tr_size = tr_x.shape[0]
            cutoff = self.construct_model(tr_x, te_x, tr_y, te_y, va_x, va_y, tr_size)
            if self.use_cutoff:
                self.logger.info('Cuttof process started')
                self.k, err = self.construct_cutoff(tr_x, tr_y, va_x, va_y, va_x.shape[0], cutoff)
                print(self.k, err)
                self.logger.info('K-cutoff: ' + self.k + ' Error:' + err)

        self.average_results()
        self.logger.info('Final validation error:' + self.va_error)



        self.final_results()
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
    def get_weights_uninet(self, input):
        weights = {
            'h1': tf.Variable(tf.truncated_normal([input, self.n_hidden_uni], 0, 0.1)),
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


    def main_effects_net_construct(self):
        me_nets = []
        for x_i in range(self.num_input):
            me_net = self.individual_univariate_net(tf.expand_dims(self.X[:, x_i], 1), self.get_weights_uninet(1),
                                                    self.get_biases_uninet())
            me_nets.append(me_net)
        return me_nets


    # Construct the model
    def construct_model(self, tr_x, te_x, tr_y, te_y, va_x, va_y, tr_size):
        # Construct model
        net = self.normal_neural_net(self.X, self.weights, self.biases)
        # check main effects need
        if self.use_main_effect_nets:
            net = net + sum(self.main_effects_net_construct())

        # Define optimizer
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
        sess.run(tf.global_variables_initializer())

        print('Initialized')
        self.logger.info('Initialized')

        for epoch in range(self.num_epochs):

            batch_order = list(range(n_batches))
            np.random.shuffle(batch_order)

            for i in batch_order:
                batch_x = tr_x[i * self.batch_size:(i + 1) * self.batch_size]
                batch_y = tr_y[i * self.batch_size:(i + 1) * self.batch_size]
                _, lr = sess.run([optimizer, decaying_learning_rate], feed_dict={self.X: batch_x, self.Y: batch_y})

            va_error = 0
            if (epoch + 1) % 50 == 0:
                print('Epoch', epoch + 1)
                self.logger.info('Epoch: ' + str(epoch + 1))
                print('\t', 'learning rate', lr)
                self.logger.info('\tlearning rate:' + str(lr))
                if self.is_classification:
                    # Test model
                    pred = tf.nn.sigmoid(net) if self.num_output == 2 else tf.nn.softmax(net)# Apply softmax to logits
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))

                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    tr_acc = accuracy.eval(feed_dict={self.X: tr_x, self.Y: tr_y},session=sess)
                    va_acc = accuracy.eval(feed_dict={self.X: va_x, self.Y: va_y},session=sess)
                    te_acc = accuracy.eval(feed_dict={self.X: te_x, self.Y: te_y},session=sess)
                    print('\t', 'train acc', tr_acc, 'val acc', va_acc, 'test acc', te_acc)
                    self.logger.info('\ttrain acc:' + str(tr_acc) + ' val acc:' + str(va_acc) + ' test acc:', str(te_acc))
                    #auc
                    auc, auc_op = tf.metrics.auc(labels=self.Y, predictions=pred)
                    # acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(self.Y,1), predictions=tf.argmax(pred,1))
                    sess.run(tf.local_variables_initializer())

                    tr_auc = sess.run([auc, auc_op], feed_dict={self.X: tr_x,self.Y: tr_y})[1]
                    va_auc = sess.run([auc, auc_op], feed_dict={self.X: va_x,self.Y: va_y})[1]
                    te_auc= sess.run([auc, auc_op], feed_dict={self.X: te_x,self.Y: te_y})[1]

                    print('\t', 'train auc', tr_auc,'val auc',  va_auc, 'test auc', te_auc)
                    self.logger.info('\ttrain auc: ' + str(tr_auc) + ' val auc:' + str(va_auc), ' test auc:' + str(te_auc))
                    va_error = 1-va_auc
                else:
                    tr_mse = sess.run(loss_op, feed_dict={self.X: tr_x, self.Y: tr_y})
                    va_mse = sess.run(loss_op, feed_dict={self.X: va_x, self.Y: va_y})
                    te_mse = sess.run(loss_op, feed_dict={self.X: te_x, self.Y: te_y})
                    print('\t', 'train rmse', math.sqrt(tr_mse), 'val rmse', math.sqrt(va_mse), 'test rmse',math.sqrt(te_mse))
                    self.logger.info('\ttrain rmse:' + str(math.sqrt(tr_mse))+ ' val rmse:' + str(math.sqrt(va_mse)) + ' test rmse:' + str(math.sqrt(te_mse)))
                    va_error=math.sqrt(va_mse)

        print('done')
        self.logger.info('Done fold running')
        self.interpret_weights(sess)
        return va_error

    # Construct the cutoff model
    def construct_cutoff(self, tr_x, tr_y, va_x, va_y, size, cutoff):

        interaction_ranking = sorted(self.global_interaction_strengths.items(), key=operator.itemgetter(1),reverse=True)

        err = self.run_network(sum(self.main_effects_net_construct()), size, va_x, va_y, self.l2_norm, self.l2_const)
        k = 0

        for i in range(len(interaction_ranking)):
            interactions_uninets = []
            for j in range(i+1):
                interaction = self.get_slice_of_data(interaction_ranking[j][0])
                interactions_uninets.append(self.individual_univariate_net(interaction, self.get_weights_uninet(len(interaction_ranking[j][0])),
                                               self.get_biases_uninet()))

            # access weights & biases
            self.weights = self.create_weights()
            self.biases = self.create_biases()

            net=None
            sess=None
            loss_op= None
            if err > cutoff:
                print(interaction_ranking[i])
                self.logger.info(str(interaction_ranking[i]))
                net = sum(self.main_effects_net_construct()) + sum(interactions_uninets)
                err, sess, loss_op = self.run_network(net, size, va_x, va_y, self.l2_norm, self.l2_const)
                k += 1
            else:
                self.test_evaluation(sess,net, loss_op)
                break
        return k, err


    def run_network(self, net, size, x_d, y_d, l_norm, l_const):
        # Define optimizer
        if self.is_classification:
            if self.num_output == 2:
                loss_op = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y,logits=net)  # use this in the case of binary classification
                loss_op = tf.reduce_mean(loss_op)
            else:
                loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=net)
                loss_op = tf.reduce_mean(loss_op)
        else:
            loss_op = tf.losses.mean_squared_error(labels=self.Y, predictions=net)

        sum_l = tf.reduce_sum([l_norm(self.weights[k]) for k in self.weights])
        loss_w_reg_op = loss_op + l_const * sum_l

        batch = tf.Variable(0)
        decaying_learning_rate = tf.train.exponential_decay(self.learning_rate, batch * self.batch_size, size, 0.95, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=decaying_learning_rate).minimize(loss_w_reg_op, global_step=batch)

        n_batches = size // self.batch_size
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.25
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        print('Initialized')
        self.logger.info('Initialized')

        error_test_net = 0
        for epoch in range(self.num_epochs):

            batch_order = list(range(n_batches))
            np.random.shuffle(batch_order)

            for i in batch_order:
                batch_x = x_d[i * self.batch_size:(i + 1) * self.batch_size]
                batch_y = y_d[i * self.batch_size:(i + 1) * self.batch_size]
                _, lr = sess.run([optimizer, decaying_learning_rate], feed_dict={self.X: batch_x, self.Y: batch_y})

            if (epoch + 1) % 50 == 0:
                print('Epoch', epoch + 1)
                self.logger.info('Epoch: ' + str(epoch + 1))
                print('\t', 'learning rate', lr)
                self.logger.info('\tlearning rate:' + str(lr))

                if self.is_classification:
                    # Test model
                    pred = tf.nn.sigmoid(net) if self.num_output == 2 else tf.nn.softmax(net)  # Apply softmax to logits
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))

                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    acc_va = accuracy.eval(feed_dict={self.X: x_d, self.Y: y_d}, session=sess)
                    print('\t', 'acc validation', acc_va)
                    self.logger.info('\tacc acc: ' + str(acc_va))

                    # auc
                    auc, auc_op = tf.metrics.auc(labels=self.Y, predictions=pred)
                    sess.run(tf.local_variables_initializer())
                    error_validation = sess.run([auc, auc_op], feed_dict={self.X: x_d, self.Y: y_d})[1]
                    print('\t', 'auc validation', error_validation)
                    self.logger.info('\tauc validation: ' + str(error_validation))

                    error_test_net = 1-error_validation
                else:
                    mse_validation = sess.run(loss_op, feed_dict={self.X: x_d, self.Y: y_d})
                    rmse_validation =  math.sqrt(mse_validation)
                    print('\t', 'rmse validation', rmse_validation)
                    self.logger.info('\trmse validation: ' + str(rmse_validation))

                    error_test_net = rmse_validation

        return error_test_net, sess, loss_op


    def test_evaluation(self, sess, net, te_x, te_y, loss_op):
        if self.is_classification:
            # Test model
            pred = tf.nn.sigmoid(net) if self.num_output == 2 else tf.nn.softmax(net)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))

            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            acc_te = accuracy.eval(feed_dict={self.X: te_x, self.Y: te_x}, session=sess)
            print('\t', 'acc test', acc_te)
            self.logger.info('\tacc test: ' + str(acc_te))

            # auc
            auc, auc_op = tf.metrics.auc(labels=self.Y, predictions=pred)
            sess.run(tf.local_variables_initializer())
            error_test = sess.run([auc, auc_op], feed_dict={self.X: te_x, self.Y: te_y})[1]
            print('\t', 'auc test', error_test)
            self.logger.info('\tauc test: ' + str(error_test))

            self.error_test_net += (1 - error_test)
        else:
            mse_test = sess.run(loss_op, feed_dict={self.X: te_x, self.Y: te_y})
            rmse_test = math.sqrt(mse_test)
            print('\t', 'rmse test', rmse_test)
            self.logger.info('\trmse test: ' + str(rmse_test))

            self.error_test_net += rmse_test

    def get_slice_of_data(self,interaction):
        features = []
        for feature in interaction:
            features.append(tf.expand_dims(self.X[:, feature-1],1))
        return tf.concat(features,1)


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

        heatmap_df = pd.DataFrame(np.array(pairwise_2d))
        heatmap_df.index += 1
        heatmap_df.columns += 1
        #print heatmap
        sns.set()
        self.ax = sns.heatmap(heatmap_df, cmap='Blues',cbar_kws={"shrink": .5})
        # save heatmap as file
        if os.path.isfile(self.heatmap_name):
            os.remove(self.heatmap_name)
        plt.savefig(self.heatmap_name)
        # plt.show()


    def interpret_weights(self, sess):
        w_dict = sess.run(self.weights)

        # Higher-Order Interaction Ranking
        interaction_ranking = self.get_interaction_ranking(w_dict)
        print('\nHigher-Order Interaction Ranking')
        print(interaction_ranking)
        self.logger.info('Higher-Order Interaction Ranking\n' + str(interaction_ranking))

        # Pairwise Interaction Ranking
        pairwise_ranking = self.get_pairwise_ranking(w_dict)
        print('\nPairwise Interaction Ranking')
        print(pairwise_ranking)
        self.logger.info('Pairwise Interaction Ranking\n' + str(pairwise_ranking))

    # final results
    def average_results(self):
        for pair, cab in self.global_pairwise_strengths.items():
            self.global_pairwise_strengths[pair] = float(cab) / self.k_fold

        for interaction, cab in self.global_interaction_strengths.items():
            self.global_interaction_strengths[interaction] = float(cab) / self.k_fold

        self.error_test_net =  float(self.error_test_net) / self.k_fold



    def final_results(self):
        pairwise_ranking = sorted(self.global_pairwise_strengths.items(), key=operator.itemgetter(1), reverse=True)
        interaction_ranking = sorted(self.global_interaction_strengths.items(), key=operator.itemgetter(1), reverse=True)
        print('\nFinal results:', '\n##############')
        print('\nHigher-Order Interaction Ranking')
        print(interaction_ranking)
        self.logger.info('Final results:\n##############\nHigher-Order Interaction Ranking\n' + str(interaction_ranking))
        print('\nPairwise Interaction Ranking')
        print(pairwise_ranking)
        self.logger.info('Pairwise Interaction Ranking\n' + str(pairwise_ranking))
        self.write_to_csv(interaction_ranking, self.higher_order_out_name)
        self.write_to_csv(pairwise_ranking, self.pairwise_out_name)


    def write_to_csv(self, interactions, name):
        with open(name , 'w') as out:
            csv_out = csv.writer(out, lineterminator='\n')
            csv_out.writerow(['Features', 'Strength'])
            for row in interactions:
                csv_out.writerow(row)


