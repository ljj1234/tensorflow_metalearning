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
from itertools import  product
import math
from sklearn.random_projection import SparseRandomProjection

#test_dict = defaultdict(lambda:'')
#for x in test_list:
#    pair_list = x.split(':')
#    test_dict[pair_list[0]] = pair_list[1]
##global variable    
all_label_result = defaultdict(lambda:set())
high_label_result = defaultdict(lambda:list())
feature_length_result = defaultdict()
total_length = 0
##including high key now
key_list = ['begid','city_code','doc_category','doc_original','interestbizs','age','pic_num','province_code','sex','subcategories','usercategories','video_num','vulgar','direction']

all_label_encoder = defaultdict(lambda:[])
with open('../../austinyang/data/cross_feature_list.txt') as file:
    cross_lines = file.readlines()
high_key_list = ['ttseg_hash','kt_hash']
total_list = key_list+high_key_list
test = None
global_gen_data = []
def csr_vappend(a,b):
#""" Takes in 2 csr_matrices and appends the second one to the bottom of the first one. Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites the first matrix instead of copying it. The data, indices, and indptr still get copied."""
    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a
def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)

def generate_key_value_data(file_list): 
    data_result = []
    data_dict = defaultdict(lambda:'')
    for line in file_list:
        data_dict = defaultdict(lambda:set())
        line_list = line.split('\t')
        data_dict['label'] = line_list[0]
        for x in line_list[1::]:
            x_pair = x.split(':')
            data_dict[x_pair[0]].add(x_pair[1])
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
            if x_pair[0] == 'kt_hash' or x_pair[0] == 'ttseg_hash':
                high_label_result[x_pair[0]].append(x_pair[1])

def control_feature_length(cut_num):
    print('start count')
    for key,lists in high_label_result.items():
        high_dict = Counter(lists)
        high_list = list(pair[0] for pair in high_dict.most_common(cut_num))
        all_label_result[key] = high_list
        print('high length',len(all_label_result[key])) 

##one hot label class get
#@profile
def get_all_onehot(feature,label_list):
    global total_length
    print('start encode feature: ',feature,'length: ',len(label_list))
    feature_length_result[feature] = len(label_list)
    total_length += len(label_list)
    #print('one rank dim',total_length)
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
#@profile
def get_encode_result(all_label_encoder,feature_list,data_input):
    result = defaultdict(lambda:'')
    for feature in feature_list:
        data = list(data_input[feature])
        #print(data)
        enc = all_label_encoder[feature][0]
        #one_hot = all_label_encoder[feature][1]
        enc_class = all_label_encoder[feature][1]
        data = [data_value for data_value in data if data_value in set(all_label_result[feature])]
        if len(data) != 0 :
            new_data = enc.transform(data)
            onehot_index_array = np.intersect1d(enc_class,new_data,True)
            #print(new_data)
            #new_data = new_data.reshape(-1,1)
            #new_data_result = one_hot.transform(new_data)
            #new_data_result = np.sum(new_data_result,axis = 0)
            #print type(new_data_result)
            #pdb.set_trace()
            result[feature]= onehot_index_array
        else:
            #print('no_data',feature)
            result[feature] = np.array([])

    return result
##if control length then do it
def get_high_encode_result(all_label_encoder,feature_list,data_input):
    result = defaultdict(lambda:'')
    for feature in feature_list:
        data = list(data_input[feature])
        #print(data)
        enc = all_label_encoder[feature][0]
        #one_hot = all_label_encoder[feature][1]
        data = [data_value for data_value in data if data_value in set(all_label_result[feature])]
        #if data and all(data_value in set(all_label_result[feature]) for data_value in data):
        enc_class = all_label_encoder[feature][1]
        if len(data) != 0:
            new_data = enc.transform(data)
            onehot_index_array = np.intersect1d(enc_class,new_data,True)
            #print(new_data)
            #new_data = new_data.reshape(-1,1)
            #new_data_result = one_hot.transform(new_data)
            #new_data_result = np.sum(new_data_result,axis = 0)
            #print type(new_data_result)
            result[feature]= onehot_index_array
        else:
            #print('no_data',feature)
            result[feature] = np.array([])

    return result

#@profile
def get_cross_feature(feature_a, feature_b,one_hot_data_result): 
    #pdb.set_trace()
    return list(product(one_hot_data_result[feature_a],one_hot_data_result[feature_b]))


#@profile
def multi_process_instance(test_data_index):
    #pdb.set_trace() 
    test_data = global_gen_data[test_data_index]
    one_hot_data_result = get_encode_result(all_label_encoder,key_list,test_data)
    one_hot_high_data_result = get_high_encode_result(all_label_encoder,high_key_list,test_data)
    #final_result = csr_matrix(np.array([]))

    #for feature,result in one_hot_data_result.items():
    #    #print(np.sum(sparse_result.toarray(),axis = 0))
    #    #print(sparse_result.todense())
    #    if result is not '' : 
    #        #array_result = np.sum(result.toarray(),axis = 0)
    #        array_result = csr_matrix(result.sum(axis=0))
    #        #print(len(array_result))
    #        #final_result = np.append(final_result, array_result)
    #        final_result = hstack((final_result,array_result),format = 'csr')
    #    else:
    #        #final_result = np.append(final_result, np.zeros(len(all_label_result[feature])))
    #        final_result = hstack((final_result, csr_matrix(np.zeros(len(all_label_result[feature])))),format = 'csr')
    #        one_hot_data_result[feature] = csr_matrix(np.zeros(len(all_label_result[feature])))

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
    one_hot_data_result.update(one_hot_high_data_result)
    cross_result = defaultdict()
    for line in cross_lines:
        cross_feat = line.strip().split()
        feat_a = cross_feat[0]
        feat_b = cross_feat[1]
        #print feat_a, ' x ', feat_b
        cross_result[line] = get_cross_feature(feat_a, feat_b,one_hot_data_result)
    final_result = []
    position_num = 0
    for feature,result in one_hot_data_result.items():
        #print(np.sum(sparse_result.toarray(),axis = 0))
        #print(sparse_result.todense())
        if feature == 'sex':
            continue
        for one_index in result:
            final_result.append(position_num+one_index)
        position_num += feature_length_result[feature]
    for feature_line,result in cross_result.items():
        cross_feat = feature_line.strip().split()
        feat_a = cross_feat[0]
        feat_b = cross_feat[1]
        for coordinate in result:
            #if 1 in coordinate:
            final_result.append(position_num+(coordinate[0]*feature_length_result[feat_b] +coordinate[1]))
        position_num += feature_length_result[feat_a]*feature_length_result[feat_b]

    for one_index in one_hot_data_result['sex']:
        final_result.append(position_num+one_index)
    position_num += feature_length_result['sex']
        
            #array_result = np.sum(result.toarray(),axis = 0)
            #array_result = csr_matrix(result.sum(axis=0))
            #print(len(array_result))
            #final_result = np.append(final_result, array_result)
            #final_result = hstack((final_result,array_result),format = 'csr')
       # else:
       #     #final_result = np.append(final_result, np.zeros(len(all_label_result[feature])))
       #     final_result = hstack((final_result, csr_matrix(np.zeros(len(all_label_result[feature])))),format = 'csr')
       #     one_hot_data_result[feature] = csr_matrix(np.zeros(len(all_label_result[feature])))
    #for cross_data in cross_result:
        #cross_data.toarray()
        #final_result = np.append(final_result, cross_data.toarray().reshape(-1))
        #final_result = hstack((final_result, csr_matrix(cross_data.toarray().reshape(-1))),format = 'csr')
#    if data_result_chunk is not '':
#        data_result_chunk = vstack((data_result_chunk,final_result),format='csr')
#    else:
#        data_result_chunk = final_result
#    if count_feq == 1: 
#        print('check shape', data_result_chunk.shape)
    #pdb.set_trace()
    #print('position_num',position_num)
    return final_result

#@profile
def main():
    global global_gen_data
    global total_length
    with open('feature_select_list.pkl','r') as f:
        feature_select_list = pickle.load(f)
    #pdb.set_trace()
    cores = multiprocessing.cpu_count()
    #21
    for file_number in xrange(1):
        with open('../order_100_data/order_data_chunk_' + str(file_number), 'r') as f:
            file_list = f.readlines() 
            print('read done:' + str(file_number))
            get_all_label(file_list)
#    cores = multiprocessing.cpu_count()
#    pool = multiprocessing.Pool(processes=(cores-2))

    #pdb.set_trace()
    #print('length: ',len(all_label_result['usercategories']))
    cut_num = 2000
    control_feature_length(cut_num)
    #save_pickle(all_label_result,'all_label.pkl')
    #pdb.set_trace()
    for feature in total_list:
        enc, one_hot = get_all_onehot(feature,list(all_label_result[feature]))
        all_label_encoder[feature].extend([enc,one_hot])
   # rewards = []
   # items_id = []
   # uin = []
   # for file_number in range(2,16): 
   #     with open('../order_100_event_data/order_data_id_label_chunk_' + str(file_number), 'r') as f:
   #         file_list = f.readlines()
   #         #pdb.set_trace()
   #         for line in file_list:
   #             line_list = line.split('\t')
   #             #if len(line_list) < 3:
   #                 #print(line_list)
   #             rewards.append(line_list[1])
   #             items_id.append(line_list[0])
   #             uin.append(line_list[2].strip('\n'))

    for line in cross_lines:
        cross_feat = line.strip().split()
        feat_a = cross_feat[0]
        feat_b = cross_feat[1]
        total_length += (feature_length_result[feat_a] * feature_length_result[feat_b])

    srp = SparseRandomProjection(n_components=1000)
    print('total_d_length',total_length)
    for file_number in xrange(0, 4):
        rewards = []
        items_id = []
        uin = []
        with open('../order_new_pool_data/order_data_id_label_chunk_' + str(file_number), 'r') as f:
            file_list = f.readlines()
            #pdb.set_trace()
            for line in file_list:
                line_list = line.split('\t')
                #if len(line_list) < 3:
                    #print(line_list)
                rewards.append(line_list[1])
                items_id.append(line_list[0])
                uin.append(line_list[2].strip('\n'))
        with open('../order_new_pool_data/order_data_chunk_' + str(file_number), 'r') as f:
            file_list = f.readlines()
            #pdb.set_trace()
            gen_data = generate_key_value_data(file_list)
        with open('../order_new_pool_data/length_chunk_'+str(file_number),'r') as f:
            cut_pool_list = pickle.load(f)
        #gen_data = gen_data[0:100]
        print('start file: ' + str(file_number))
        print('number chunk',len(cut_pool_list)/4000)
        chunk_file_number = len(cut_pool_list)/4000
        pdb.set_trace()
        cut_start_flag = 0
        for block_num in range(chunk_file_number):
            print('-------------------------------')
            print('strat block: ' + str(block_num+1))
            cut_pool = cut_pool_list[block_num*4000:(block_num+1)*4000]
            cut_end = sum(cut_pool)
            print('chunk_range: ',cut_start_flag,cut_end+cut_start_flag)
            data_todeal = gen_data[cut_start_flag:(cut_end+cut_start_flag)]
            rewards_todeal = rewards[cut_start_flag:(cut_end+cut_start_flag)]
            items_todeal = items_id[cut_start_flag:(cut_end+cut_start_flag)]
            uin_todeal = uin[cut_start_flag:(cut_end+cut_start_flag)]
            cut_start_flag += cut_end
            pdb.set_trace()
            #multi
#            global_gen_data.extend(data_todeal)
#            pool = multiprocessing.Pool(processes=(cores-4))
#            cnt = 0
#            #data_result_chunk = [] 
#            data_all = 0
#            row_chunk = np.array([])
#            col_chunk = np.array([])
#            onehot_all = []
#            indptr = [0]
#            indices = []
#            data = []
#            #debug_data_length = []
#            for y in pool.imap(multi_process_instance,range(len(global_gen_data))):
#                #data_result_chunk = y
#                #pdb.set_trace()
#                #debug_data_length.append(y)
#                for num_index in y:
#                    indices.append(num_index)
#                    data.append(1)
#                indptr.append(len(indices))
#                #onehot_total = len(data_result_chunk)
#                #onehot_all.append(onehot_total)
#                #col_chunk = np.concatenate((col_chunk,data_result_chunk))
#                #data_all += onehot_total
#                sys.stdout.write('done %d/%d\r' % (cnt, len(global_gen_data)))
#                cnt += 1
#            pool.close()
#            pool.join()
#            #pdb.set_trace()
#            save_chunk = csr_matrix((data, indices, indptr),dtype=int)                
#            save_chunk = save_chunk/math.sqrt(140)
#            #feature_select_save_chunk = save_chunk
#            #feature_select_save_chunk[:,feature_select_list] = 0
#            #feature_select_save_chunk = feature_select_save_chunk/math.sqrt(140)
#            #save_pickle(save_chunk,'../analyse_data/100d_event_data' + 'file_' + str(file_number)+ '_'+str(block_num)+'.pkl')
#            #pdb.set_trace()
#            if file_number == 0 and block_num == 0:
#                rp_save_chunk = srp.fit_transform(save_chunk)
#                save_pickle(srp,'RandomSparseProjection_new_pool.pkl')
#            else:
#                rp_save_chunk = srp.transform(save_chunk)
#            #data_all = np.sum(np.array(onehot_all))
#            #for i, one_hot in enumerate(onehot_all):
#                #row_chunk = np.concatenate((row_chunk,np.array([i]*onehot_total)))
#            #data_result_chunk = []
#            #for data in gen_data:
#            #    data_result_chunk.append(multi_process_instance(data)) 
#            #data_all = 0
#            #for row in range(len(data_result_chunk)):
#            #    onehot_total = len(data_result_chunk[row])
#            #    row_chunk = np.append(row_chunk,np.array([row]*onehot_total))
#            #    col_chunk = np.append(col_chunk,np.array(data_result_chunk[row]))
#            #    data_all += onehot_total
#            
#        #data_result_chunk = []
#        #for data in gen_data:
#        #    data_result_chunk.append(multi_process_instance(data)) 
#        #save_chunk = data_result_chunk[0]
#        #for num in range(1,len(data_result_chunk)):
#        #    save_chunk = vstack((save_chunk,data_result_chunk[num]),format='csr')
#        #pdb.set_trace()
#            event = []
#            #pdb.set_trace()
#            event_cut_flag = 0
#            for num in cut_pool:
#                save_dict = {}
#                #data_to_save = data_result_chunk[num*10:(num+1)*10]
#                #y_to_save = rewards_todeal[num*10:(num+1)*10]
#                #items_to_save = items_todeal[num*10:(num+1)*10]
#                #uin_to_save = uin_todeal[num*10:(num+1)*10]
#
#                item_dict = defaultdict()
#                rewards_dict = defaultdict()
#                rp_item_dict = defaultdict()
#                for index_num in range(num):
#                    item_dict[items_todeal[event_cut_flag+index_num]] = save_chunk[event_cut_flag+index_num]
#                    rewards_dict[items_todeal[event_cut_flag+index_num]] = rewards_todeal[event_cut_flag+index_num]
#                    rp_item_dict[items_todeal[event_cut_flag+index_num]] = rp_save_chunk[event_cut_flag+index_num]
#                save_dict['context'] = item_dict
#                save_dict['rewards'] = rewards_dict
#                save_dict['user_id'] = uin_todeal[event_cut_flag] 
#                save_dict['rp_context'] = rp_item_dict
#                event.append(save_dict) 
#                event_cut_flag += num
#            #save_chunk = csr_matrix((np.array([1]*data_all), (row_chunk, col_chunk)), shape=(20000, total_length))
#            #fill_chunk = csr_matrix((20000, total_length-save_chunk.shape[1]), dtype=int)
#            #save_chunk = csr_vappend(save_chunk,fill_chunk)
#            #save_chunk = hstack((save_chunk,np.zeros((20000,total_length-save_chunk.shape[1]))),format='csr')
#            #pdb.set_trace()
#            save_pickle(event,'../new_pool_event_data/100d_event_data' + 'file_' + str(file_number)+ '_'+str(block_num)+'.pkl')
#            global_gen_data = []
           
        #for block_num, data_result_chunk in enumerate(chunck_result_list):
        #    print('save_pickle')
        #    save_pickle(data_result_chunk,'../all_multi_data/all_4061d_data' + 'file_' + str(file_number)+ '_chunk_'+str(block_num)+'.pkl')

if __name__ == '__main__':
    #cProfile.run('main()')
    main()
