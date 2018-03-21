
from __future__ import print_function
from __future__ import division
from collections import defaultdict 


import tensorflow as tf
import pickle
import numpy as np
import heapq

# Parameters
learning_rate = 3e-2
training_epochs = 1
display_step = 1

# Network Parameters
n_hidden_1 = 100 # 1st layer number of neurons
n_hidden_2 =  100# 2nd layer number of neurons
#change here for data dims
n_input = 100 # data input dims(shape: 1*50)

n_classes = 2 # total classes 
windows_size = 3 #average

class LWA_model:

    def __init__(self):


        self.name = 'LWA model'
        self.step = 0
        # self.rewards_optimal = []
        self.rewards = []
        self.event_history = defaultdict(lambda: defaultdict(lambda: None))
        self.sess = tf.Session()
        self.max_key = 0
        self.total_cost = 0.0
        self.biuld_model()
        self.f_shape = [n_hidden_1, n_hidden_2]

        def biuld_model(self):
            # tf Graph input
            self.X = tf.placeholder("float", [None, n_input])
            self.Y = tf.placeholder("float", [None])
            self.postive_input = tf.placeholder("float", [None, windows_size, n_input])
            self.negative_input = tf.placeholder("float",[None, windows_size, n_input])

            p_input_reshape = tf.reshape(self.postive_input,[-1,n_input])
            n_input_reshape = tf.reshape(self.negative_input,[-1,n_input])

            # Create model
            def multilayer_perceptron(self,x_layers, p_layers, n_layers):

                for i in xrange(len(self.f_shape)-1):
                    with tf.name_scope("f_hidden%d" % (i+1)):
                        w = tf.Variable(tf.truncated_normal([self.f_shape[i], self.f_shape[i+1]], 
                            stddev=1.0/math.sqrt(float(self.f_shape[i]))), name='weights')
                        b = tf.Variable(tf.zeros([self.f_shape[i+1]]), name="bias")
                        p_layers = tf.nn.relu(tf.matmul(p_layers, w) + b)
                        n_layers = tf.nn.relu(tf.matmul(n_layers, w) + b)
                        x_layers = tf.nn.relu(tf.matmul(x_layers, w) + b)

                return x_layers, p_layers, n_layers

            def get_length(self, sequence):
                used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
                length = tf.reduce_sum(used, 1)
                length = tf.cast(length, tf.float32)
                length = tf.clip_by_value(length, 1, windows_size)
                return tf.reshape(length, [-1, 1])


            def exp_weight(self, sequence):
                l = self.length(sequence)
                weight_range = l - 1 - tf.range(windows_size, dtype=tf.float32)
                weight = tf.exp(-0.1 * weight_range)
                weight = tf.clip_by_value(weight, 0, 1)

                sum_weight = tf.reduce_sum(weight, 1) - (windows_size - tf.reshape(l, [-1]))

                return tf.reshape(weight, [-1, windows_size, 1]), tf.reshape(sum_weight, [-1, 1])


            def aggregating_function(self, p_repre, n_repre):

                pos_hidden_r = tf.reshape(p_repre, [-1, self.max_k, self.f_shape[-1]])
                neg_hidden_r = tf.reshape(n_repre, [-1, self.max_k, self.f_shape[-1]])

                x_embeddings = x_hidden
                self.x_embeddings_size = tf.shape(x_embeddings)

                pos_length = self.length(pos_hidden_r)
                neg_length = self.length(neg_hidden_r)

                pos_weight, pos_sum_weight = self.exp_weight(pos_hidden_r)
                neg_weight, neg_sum_weight = self.exp_weight(neg_hidden_r)

                #pos_embeddings = tf.reduce_sum(pos_hidden_r, 1) / pos_length
                #neg_embeddings = tf.reduce_sun(neg_hidden_r, 1) / neg_length
                postive_final_repre = tf.divide(tf.reduce_sum(pos_hidden_r, 1), pos_length)
                negative_final_repre = tf.divide(tf.reduce_sum(neg_hidden_r, 1), neg_length)

                return positive_final_repre, negative_final_repre

            def final_weight(po_items,na_items):
                postive_wight = tf.Variable([0.5], dtype=tf.float32, name="w1")
                negative_wight = tf.Variable([0.5], dtype=tf.float32, name="w0")
                theta = tf.add((negative_wight * na_items),(postive_wight * po_items))
                return theta

            def get_logits(test_items,final_weight):
                # return tf.matmul(test_items,(tf.add(tf.matmul(na_items,weights['w0']),tf.matmul(po_items,weights['w1'])))) + biases['b_pre']
                # return tf.add(tf.matmul(na_items,weights['w0']),tf.matmul(po_items,weights['w1'])) + biases['b_pre']
                bias = tf.Variable([.0], dtype=tf.float32, name="logtis-bias")
                return tf.reduce_sum(tf.multiply(test_items,final_weight), 1) + bias


            # Construct model
            x_repre, p_repre, n_repre = multilayer_perceptron(self.X,p_input_reshape,n_input_reshape)
            pos_embeddings,neg_embeddings = aggregating_function(p_repre, n_repre)
            final_weight = final_weight(pos_embeddings, neg_embeddings)
            self.logits = get_logits(x_repre,final_weight)
            # print('logits shape',self.logits)
            self.logits_pred = tf.sigmoid(self.logits)
            # Define loss and optimizer
            self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

            # self.pred = tf.nn.softmax(logits)
            # Initializing the variables
            self.init = tf.global_variables_initializer()
            self.sess.run(init)

    def take_action(self,event): 

        #Save history
        user_id = event['user_id']
        postive_data = self.event_history[user_id]['p']
        negative_data = self.event_history[user_id]['n']
        rewards_dict = event['rewards']
        for item_id, context in event['context']:
            if rewards_dict[item_id] == 1:
                postive_data = self.event_history[user_id]['p']
                if postive_data:
                    self.event_history[user_id]['p'] = np.vstack(postive_data,context)
                else:
                    self.event_history[user_id]['p'] = context
            else:
                negative_data = self.event_history[user_id]['n']
                if negative_data:
                    self.event_history[user_id]['n'] = np.vstack(negative_data,context)
                else:
                    self.event_history[user_id]['n'] = context

        #pick newest K p&n data
        user_data_p = self.event_history[user_id]['p']
        user_data_n = self.event_history[user_id]['n']

        #postive class data
        if user_data_p: 
            if user_data_p.shape[0] < windows_size:
                user_data_p = np.vstack(user_data_p,np.zeros(windows_size - user_data_p.shape[0],n_input))
            p_input = user_data_p[-windows_size:]
        else:
            p_input = np.zeros([windows_size,input])

        #negative class data
        if user_data_n: 
            if user_data_n.shape[0] < windows_size:
                user_data_n = np.vstack(user_data_n,np.zeros(windows_size - user_data_n.shape[0],n_input))
            n_input = user_data_n[-windows_size:]
        else:
            n_input = np.zeros([windows_size,input])

        p_list = []
        choice = []
        for item_id, context in event['context']:
            p = self.sess.run(self.logits_pred,feed_dict={self.X:context,
                p_input,n_input})
            p_list.append(p)
            choice_list.append(item_id)
        sorted_list = heapq.nlargest(5, p_list)
        for x in sorted_list:
            choice.append(choice_list[p_list.index(x)])
        return choice


    def update(self, event, choice):

        context = event['context'][choice]
        reward = event['rewards'][choice]
        user_id = event['user_id']
        self.step += 1
        self.rewards.append(reward)

        #pick newest K p&n data
        user_data_p = self.event_history[user_id]['p']
        user_data_n = self.event_history[user_id]['n']

        #postive class
        if user_data_p: 
            if user_data_p.shape[0] < windows_size:
                user_data_p = np.vstack(user_data_p,np.zeros(windows_size - user_data_p.shape[0],n_input))
            p_input = user_data_p[-windows_size:]
        else:
            p_input = np.zeros([windows_size,input])

        #negative class
        if user_data_n: 
            if user_data_n.shape[0] < windows_size:
                user_data_n = np.vstack(user_data_n,np.zeros(windows_size - user_data_n.shape[0],n_input))
            n_input = user_data_n[-windows_size:]
        else:
            n_input = np.zeros([windows_size,input])

        _, c = self.sess.run([self.optimizer,self.cost], feed_dict={self.X: context,
            self.Y: reward, p_input, n_input})

        self.total_cost += c
        avg_cost = self.total_cost / self.step
        print ("cost={:.9f}".format(avg_cost))
        
        # self.rewards_optimal.append(event['op_ra'])


    def predict(self, pos_history, neg_history, context):
        pred_sigmoid = self.sess.run(self.logits_pred, {self.pos_cand:pos_history, self.neg_cand:neg_history, self.x:context})

        # print "%s AUC:%f" % (key, roc_auc_score(pred_labels, pred_sigmoid))
        return pred_sigmoid
        

    def get_info(self):
        '''Return model information'''

        return self.name, self.step, self.rewards

        




















