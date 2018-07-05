# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import time
import sys
import pdb
import pickle
from os import listdir
from scipy import sparse
from sklearn.metrics import roc_auc_score
import multiprocessing as mp
from utils import to_libsvm_file
import gzip


if __name__ == '__main__':
    
    print 'Load data...' 
    X = None
    y = np.array([])
    

    for file_number in range(0,2):
        for chunk in range(0,20):
            #print 'Chunk:', chunk
            X_chunk= pickle.load(gzip.open('../data/data_100_v2/50d_datafile_' +str(file_number)+'_'+ str(chunk) + '.pkl.gz', 'rb'))
            y_chunk = pickle.load(open('../data/data_100/new_order_data_labelfile_'+str(file_number)+'_chunk_'+ str(chunk) + '.pkl', 'rb'))
            #print(type(y_chunk[0]))
            
            if X is None:
                X = X_chunk
            else:
                X = sparse.vstack([X, X_chunk], format='csr')

            y = np.append(y, y_chunk)
    to_libsvm_file(X,y,'dnn_libsvm_train_file')
    pdb.set_trace()

    for datafile in range(0,2):
        for i in range(0,20):
            #print 'Chunk:', chunk
            stream_i = pickle.load(open('../data/100w_ftrl_100_ucb/100d_event_datafile_' + str(datafile) + '_' + str(i) + '.pkl'))
            #print(type(y_chunk[0]))
            embed_stream = []
            for event in stream_i:
                items_list = event['context'].keys()
            
    time_start = time.time()
    amount_block = 13
    interval_block = 1
    amount_chunk = 10
    # for block in range(amount_block):
    for block in range(2, amount_block):
        print '*' * 80
        print 'Block:', block
        print '=' * 50
        print 'Loading pickles...'
        stream = np.array([])
        for datafile in range(block * interval_block, (block + 1) * interval_block):
            for i in range(amount_chunk):
            # for i in range(1):
                print 'Datafile:', datafile, ', chunk:', i
                # stream_i = pickle.load(open('../data/data_event_new_pool_select/100d_event_datafile_' + str(datafile) + '_' + str(i) + '.pkl'))
                stream_i = pickle.load(open('../data/100w_ftrl_100_ucb/100d_event_datafile_' + str(datafile) + '_' + str(i) + '.pkl'))
                # stream_i = pickle.load(open('../data/new_pool_keep_event_data/100d_event_datafile_' + str(datafile) + '_' + str(i) + '.pkl'))
                stream = np.append(stream, stream_i)
        # stream = stream[:100]
        print 'Stream length:', len(stream)
        # pdb.set_trace()
    
