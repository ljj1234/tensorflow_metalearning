# coding: utf-8
import sys
from collections import defaultdict
from scipy.sparse import csr_matrix,vstack,hstack
import operator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import cPickle as pickle
import cProfile
import pdb
from collections import Counter
import time
import multiprocessing

#test_dict = defaultdict(lambda:'')
#for x in test_list:
#    pair_list = x.split(':')
#    test_dict[pair_list[0]] = pair_list[1]
##global variable    
all_label_result = defaultdict(lambda:set())
high_label_result = defaultdict(lambda:[])
##including high key now
key_list = ['begid','city_code','doc_category','doc_original','interestbizs','age','pic_num','province_code','sex','subcategories','usercategories','video_num','vulgar','direction','ttseg_hash','kt_hash']

all_label_encoder = defaultdict(lambda:[])
with open('../../austinyang/data/cross_feature_list.txt') as file:
    cross_lines = file.readlines()
high_key_list = ['ttseg_hash','kt_hash']
total_list = key_list
test = None
global_gen_data = []
def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)

def generate_key_value_data(file_list): 
    data_result = []
    data_dict = defaultdict(lambda:'')
    for line in file_list:
        data_dict = defaultdict(lambda:[])
        line_list = line.split('\t')
        data_dict['label'] = line_list[0]
        for x in line_list[1::]:
            x_pair = x.split(':')
            data_dict[x_pair[0]].append(x_pair[1])
        data_result.append(data_dict)
    return data_result

#print('finish data read,get all data result')

def get_all_label(file_list): 
    data_result = []
    data_dict = defaultdict(lambda:'')
    for line in file_list:
        data_dict = defaultdict(lambda:[])
        line_list = line.split('\t')
        for x in line_list[1::]:
            x_pair = x.split(':')
            all_label_result[x_pair[0]].add(x_pair[1])
            #if x_pair[0] == 'kt_hash' or x_pair[0] == 'ttseg_hash':
                #high_label_result[x_pair[0]].append(x_pair[1])

def control_feature_length(cut_num):
    print('start count')
    for key,lists in high_label_result.items():
        high_dict = Counter(lists)
        high_list = list(pair[0] for pair in high_dict.most_common(cut_num))
        all_label_result[key] = high_list
        print('high length',len(all_label_result[key])) 

##one hot label class get
@profile
def get_all_onehot(feature,label_list):
    print('start encode feature: ',feature,'length: ',len(label_list))
    enc = LabelEncoder()
    enc.fit(label_list)
    enc_label = enc.transform(label_list)
    #print('enc_label', enc_label)
    #print('length: ', len(enc_label))
    #enc_label = enc_label.reshape(-1,1)
    #one_hot = OneHotEncoder()
    #one_hot.fit(enc_label)
    return enc,enc_label

##encode to one hot
@profile
def get_encode_result(all_label_encoder,feature_list,data_input):
    result = defaultdict(lambda:'')
    for feature in feature_list:
        data = data_input[feature]
        #print(data)
        enc = all_label_encoder[feature][0]
        #one_hot = all_label_encoder[feature][1]
        enc_class = all_label_encoder[feature][1]
        if data:
            new_data = enc.transform(data)
            onehot_index_array = np.intersect1d(enc_class,new_data,True)
            #print(new_data)
            #new_data = new_data.reshape(-1,1)
            #new_data_result = one_hot.transform(new_data)
            #new_data_result = np.sum(new_data_result,axis = 0)
            #print type(new_data_result)
            result[feature]= one_hot_index_array
        else:
            #print('no_data',feature)
            result[feature] = ''

    return result
##if control length then do it
def get_high_encode_result(all_label_encoder,feature_list,data_input):
    result = defaultdict(lambda:'')
    for feature in feature_list:
        data = data_input[feature]
        #print(data)
        enc = all_label_encoder[feature][0]
        one_hot = all_label_encoder[feature][1]
        data = [data_value for data_value in data if data_value in set(all_label_result[feature])]
        #if data and all(data_value in set(all_label_result[feature]) for data_value in data):
        if len(data) != 0:
            new_data = enc.transform(data)
            #print(new_data)
            new_data = new_data.reshape(-1,1)
            new_data_result = one_hot.transform(new_data)
            #new_data_result = np.sum(new_data_result,axis = 0)
            #print type(new_data_result)
            result[feature]= new_data_result
        else:
            #print('no_data',feature)
            result[feature] = ''

    return result

@profile
def get_cross_feature(feature_a, feature_b,one_hot_data_result): 
    feature_a = csr_matrix(one_hot_data_result[feature_a].sum(axis=0).reshape(-1,1))
    feature_b = csr_matrix(one_hot_data_result[feature_b].sum(axis=0))
    dot_result = feature_a.dot(feature_b)
    return dot_result


@profile
def multi_process_instance(test_data_index):
    
    test_data = test_data_index
    one_hot_data_result = get_encode_result(all_label_encoder,key_list,test_data)
    #one_hot_high_data_result = get_high_encode_result(all_label_encoder,high_key_list,test_data)
    final_result = csr_matrix(np.array([]))

    for feature,result in one_hot_data_result.items():
        #print(np.sum(sparse_result.toarray(),axis = 0))
        #print(sparse_result.todense())
        if result is not '' : 
            #array_result = np.sum(result.toarray(),axis = 0)
            array_result = csr_matrix(result.sum(axis=0))
            #print(len(array_result))
            #final_result = np.append(final_result, array_result)
            final_result = hstack((final_result,array_result),format = 'csr')
        else:
            #final_result = np.append(final_result, np.zeros(len(all_label_result[feature])))
            final_result = hstack((final_result, csr_matrix(np.zeros(len(all_label_result[feature])))),format = 'csr')
            one_hot_data_result[feature] = csr_matrix(np.zeros(len(all_label_result[feature])))

#    for feature,result in one_hot_high_data_result.items():
#        #print(np.sum(sparse_result.toarray(),axis = 0))
#        #print(sparse_result.todense())
#        if result is not '' : 
#            array_result = np.sum(result.toarray(),axis = 0)
#            #print(len(array_result))
#            final_result = np.append(final_result, array_result)
#        else:
#            final_result = np.append(final_result, np.zeros(len(all_label_result[feature])))
#
#            one_hot_high_data_result[feature] = csr_matrix(np.zeros(len(all_label_result[feature])))
#
#    one_hot_data_result.update(one_hot_high_data_result)
    cross_result = []
    for line in cross_lines:
        cross_feat = line.strip().split()
        feat_a = cross_feat[0]
        feat_b = cross_feat[1]
        #print feat_a, ' x ', feat_b
        cross_result.append(get_cross_feature(feat_a, feat_b,one_hot_data_result))

    for cross_data in cross_result:
        #cross_data.toarray()
        #final_result = np.append(final_result, cross_data.toarray().reshape(-1))
        final_result = hstack((final_result, csr_matrix(cross_data.toarray().reshape(-1))),format = 'csr')
#    if data_result_chunk is not '':
#        data_result_chunk = vstack((data_result_chunk,final_result),format='csr')
#    else:
#        data_result_chunk = final_result
#    if count_feq == 1: 
#        print('check shape', data_result_chunk.shape)
    return final_result

@profile
def main():
    global global_gen_data
    cores = multiprocessing.cpu_count()
    for file_number in xrange(9):
        with open('../order_data/order_data_chunk_' + str(file_number), 'r') as f:
            file_list = f.readlines() 
            print('read done:' + str(file_number))
            get_all_label(file_list)
#    cores = multiprocessing.cpu_count()
#    pool = multiprocessing.Pool(processes=(cores-2))

    #pdb.set_trace()
    #print('length: ',len(all_label_result['usercategories']))
    #cut_num = 500
    #control_feature_length(cut_num)
    for feature in total_list:
        enc, one_hot = get_all_onehot(feature,list(all_label_result[feature]))
        all_label_encoder[feature].extend([enc,one_hot])


    for file_number in xrange(0, 1):
        with open('../order_data/order_data_chunk_' + str(file_number), 'r') as f:
            file_list = f.readlines()
            #pdb.set_trace()
            gen_data = generate_key_value_data(file_list)
        gen_data = gen_data[0:100]
        print('start file: ' + str(file_number))
        print('number chunk',len(gen_data)/20000)
        chunk_file_number = len(gen_data)/20000
        #for block_num in range(chunk_file_number):
        #    print('-------------------------------')
        #    print('strat block: ' + str(block_num+1))
        #    data_todeal = gen_data[block_num*20000:(block_num+1)*20000]
        #    #multi
        #    global_gen_data.extend(data_todeal)
            #pool = multiprocessing.Pool(processes=(cores-2))
            #cnt = 0
            #data_result_chunk = [] 
            #for y in pool.imap(multi_process_instance,range(len(global_gen_data)),chunksize=22):
            #    sys.stdout.write('done %d/%d\r' % (cnt, len(global_gen_data)))
            #    cnt += 1
            #    data_result_chunk.append(y)
            #pool.close()
            #pool.join()
            #break
        data_result_chunk = []
        for data in gen_data:
            data_result_chunk.append(multi_process_instance(data)) 
        save_chunk = data_result_chunk[0]
        for num in range(1,len(data_result_chunk)):
            save_chunk = vstack((save_chunk,data_result_chunk[num]),format='csr')
        save_pickle(save_chunk,'../test_test_data/100d_data' + 'file_' + str(file_number)+ '_'+str(block_num)+'.pkl')
        global_gen_data = []
            
        #for block_num, data_result_chunk in enumerate(chunck_result_list):
        #    print('save_pickle')
        #    save_pickle(data_result_chunk,'../all_multi_data/all_4061d_data' + 'file_' + str(file_number)+ '_chunk_'+str(block_num)+'.pkl')

if __name__ == '__main__':
    #cProfile.run('main()')
    main()
