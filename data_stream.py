# coding: utf-8
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

#test_dict = defaultdict(lambda:'')
#for x in test_list:
#    pair_list = x.split(':')
#    test_dict[pair_list[0]] = pair_list[1]
    
all_label_result = defaultdict(lambda:set())
high_label_result = defaultdict(lambda:[])
key_list = ['begid','city_code','doc_category','doc_original','interestbizs','age','pic_num','province_code','sex','subcategories','usercategories','video_num','vulgar','direction']

high_key_list = ['ttseg_hash','kt_hash']
total_list = key_list + high_key_list
test = None

def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)

#@profile
def generate_uid_key_value_data(line): 
    data_dict = defaultdict(lambda:[])
    line_list = line.split('\t')
    data_dict['label'] = line_list[0]
    for x in line_list[1::]:
        x_pair = x.split(':')
        data_dict[x_pair[0]].append(x_pair[1])
    return data_dict['uin'][0]

def generate_begid_key_value_data(line): 
    data_dict = defaultdict(lambda:[])
    line_list = line.split('\t')
    data_dict['label'] = line_list[0]
    for x in line_list[1::]:
        x_pair = x.split(':')
        data_dict[x_pair[0]].append(x_pair[1])
    return int(data_dict['begid'][0])
#print('finish data read,get all data result')

def generate_search_time_key_value_data(line): 
    data_dict = defaultdict(lambda:[])
    line_list = line.split('\t')
    data_dict['label'] = line_list[0]
    for x in line_list[1::]:
        x_pair = x.split(':')
        data_dict[x_pair[0]].append(x_pair[1])
    return int(data_dict['search_time'][0])

def generate_itemid_key_value_data(line): 
    data_dict = defaultdict(lambda:[])
    line_list = line.split('\t')
    data_dict['label'] = line_list[0]
    for x in line_list[1::]:
        x_pair = x.split(':')
        data_dict[x_pair[0]].append(x_pair[1])
    return data_dict['docid'][0]

def generate_label_key_value_data(line): 
    data_dict = defaultdict(lambda:[])
    line_list = line.split('\t')
    data_dict['label'] = line_list[0]
    for x in line_list[1::]:
        x_pair = x.split(':')
        data_dict[x_pair[0]].append(x_pair[1])
    return data_dict['docid'][0],int(data_dict['label'][0]),data_dict['uin'][0]

if __name__ == '__main__':
    total_file_list = ''
    for file_number in xrange(21):
        with open('../order_100_data/order_data_chunk_' + str(file_number), 'r') as f:
            file_list = f.readlines()
            print('read done:' + str(file_number),len(file_list))
        if total_file_list is '':
            total_file_list = file_list
        else:
            total_file_list.extend(file_list)
    print('length',len(total_file_list))
    count_index = range(len(total_file_list))
    #print(count_index)
    order_dict = {}
    for count_num in count_index:
        order_dict[count_num] = generate_uid_key_value_data(total_file_list[count_num])

    group_user_dict = defaultdict(list)
    for key,value in sorted(order_dict.iteritems()):
        group_user_dict[value].append(key)
    
    #group_user_pool = defaultdict()
    #for key,value in group_user_dict.items():
    #    group_user_pool[key] = len(value) 
    #pdb.set_trace()
    group_user_f_dict = defaultdict()
    #user_length_dict = defaultdict()
    for key,value in group_user_dict.items():
        if len(value) < 10:
            continue
        group_time_dict = {}
        for list_index in range(len(value)):
            group_time_dict[list_index] = generate_search_time_key_value_data(total_file_list[value[list_index]])
        data_ordered = sorted(group_time_dict.items(), key=lambda x: x[1])
        order_list = []
        for num in range(len(value)):
            order_list.append(value[data_ordered[num][0]])
        group_user_f_dict[key] = order_list
        #keep 50 arms event form
        if len(order_list) >= 50: 
            group_user_f_dict[key] = order_list[0:(len(order_list)/50)*50]
        #user_length_dict[key] = len(order_list)
    print('group_user_dict',len(group_user_f_dict))
    #pdb.set_trace()

    event_list = []
    pool_length_list = [] 
    for key,value in group_user_f_dict.items():
        if len(value) >= 50:
            for chunk_num in range(len(value)/50):
                event_list.append(value[chunk_num*50:(chunk_num+1)*50])
                pool_length_list.append(50)
        else:
            event_list.append(value)
            pool_length_list.append(len(value))
    event_time_dict = {}
    for event_index in range(len(event_list)):
        event_time_dict[event_index] = generate_search_time_key_value_data(total_file_list[event_list[event_index][0]])

    event_ordered = sorted(event_time_dict.items(), key=lambda x: x[1])
    event_ordered_list = []
    pool_ordered_list = []
    for num in range(len(event_list)):
        event_ordered_list.append(event_list[event_ordered[num][0]])
        pool_ordered_list.append(pool_length_list[event_ordered[num][0]])
    print('event_order',len(event_ordered_list))
    #save work
    #data_ordered = []
    #for event in event_ordered_list:
    #    data_ordered.extend(event)

            

    print('order finished')
    #print('total:',len(data_ordered))
    #pdb.set_trace()
    #print('length: ',len(all_label_result['usercategories']))
    #pdb.set_trace()
    
    #for block_num in range(len(data_ordered)/400000): 
    for block_num in range(len(event_ordered_list)/40000):
        event_todeal = event_ordered_list[block_num*40000:(block_num+1)*40000]
        length_todeal = pool_ordered_list[block_num*40000:(block_num+1)*40000]
        data_todeal = []
        for event in event_todeal:
            data_todeal.extend(event)
        #data_todeal = e[block_num*400000:(block_num+1)*400000+1]
        save_list = []
        docid_list = []
        rewards_list = []
        uin_list = []
        for num_tuple in data_todeal:
            docid,rewards,uin = generate_label_key_value_data(total_file_list[num_tuple])
            docid_list.append(docid)
            rewards_list.append(rewards)
            uin_list.append(uin)
        #pdb.set_trace()
        with open('/data/WhatRecSys/felixjjli/order_new_pool_data/order_data_id_label_chunk_' + str(block_num), 'w') as f:
            for num in range(len(docid_list)):
                f.write(str(docid_list[num]) + '\t' + str(rewards_list[num]) + '\t'+ str(uin_list[num]))
        
        for num_tuple in data_todeal:
            save_list.append(total_file_list[num_tuple])
        pdb.set_trace()
        with open('/data/WhatRecSys/felixjjli/order_new_pool_data/order_data_chunk_' + str(block_num), 'w') as f:
            for item in save_list:
              f.write(item)
        pickle.dump(length_todeal,open('/data/WhatRecSys/felixjjli/order_new_pool_data/length_chunk_'+str(block_num),'wb'))



