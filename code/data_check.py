# coding: utf-8
from collections import defaultdict
from scipy.sparse import csr_matrix
import operator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
import cProfile

#test_dict = defaultdict(lambda:'')
#for x in test_list:
#    pair_list = x.split(':')
#    test_dict[pair_list[0]] = pair_list[1]
    
all_label_result = defaultdict(lambda:set())
key_list = ['kt_hash','begid','city_code','doc_category','doc_original','interestbizs','age','pic_num','province_code','sex','subcategories','usercategories','video_num','vulgar','ttseg_hash']

#key_list = ['ttseg_hash']
test = None
#@profile
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
    return data_result[2]

#print('finish data read,get all data result')

#@profile
def get_all_label(file_list): 
    data_result = []
    data_dict = defaultdict(lambda:'')
    for line in file_list:
        data_dict = defaultdict(lambda:[])
        line_list = line.split('\t')
        for x in line_list[1::]:
            x_pair = x.split(':')
            all_label_result[x_pair[0]].add(x_pair[1])

#print('get all label list')
#@profile
def get_all_onehot(feature,label_list):
    print('start encode feature: ',feature,'length: ',len(label_list))
    enc = LabelEncoder()
    enc.fit(label_list)
    enc_label = enc.transform(label_list)
    #print('enc_label', enc_label)
    #print('length: ', len(enc_label))
    enc_label = enc_label.reshape(-1,1)
    one_hot = OneHotEncoder()
    one_hot.fit(enc_label)
    return enc,one_hot

#@profile
def get_encode_result(all_label_encoder,feature_list,data_input,all_label_result):
    result = defaultdict(lambda:'')
    for feature in feature_list:
        data = data_input[feature]
        print(data)
        enc = all_label_encoder[feature][0]
        one_hot = all_label_encoder[feature][1]
        if data:
            new_data = enc.transform(data)
            print(new_data)
            new_data = new_data.reshape(-1,1)
            new_data_result = one_hot.transform(new_data)
            #new_data_result = np.sum(new_data_result,axis = 0)
            #print type(new_data_result)
            result[feature]= new_data_result
        else:
            print('no_data',feature)
            result[feature] = ''

    return result

for file_number in xrange(10):
    with open('../../data/part-0000' + str(file_number), 'r') as f:
        file_list = f.readlines() 
    print('read done:' + str(file_number))
    if file_number == 0:
        test = generate_key_value_data(file_list)
    get_all_label(file_list)

#print('length: ',len(all_label_result['usercategories']))
all_label_encoder = defaultdict(lambda:[])
for feature in key_list:
    enc, one_hot = get_all_onehot(feature,list(all_label_result[feature]))
    all_label_encoder[feature].extend([enc,one_hot])
one_hot_data_result = get_encode_result(all_label_encoder,key_list,test,all_label_result)
final_result = np.array([])
for feature,result in one_hot_data_result.items():
    #print(np.sum(sparse_result.toarray(),axis = 0))
    #print(sparse_result.todense())
    if result is not '' : 
        array_result = np.sum(result.toarray(),axis = 0)
        #print(len(array_result))
        final_result = np.append(final_result, array_result)
    else:
        final_result = np.append(final_result, np.zeros(len(all_label_result[feature])))
ttseg_result = csr_matrix(one_hot_data_result['ttseg_hash'].sum(axis=0).reshape(-1,1))
kt_result = csr_matrix(one_hot_data_result['kt_hash'].sum(axis=0))
dot_result = ttseg_result.dot(kt_result)
print('dot shape:', dot_result.shape)
dot_result.toarray()
#final_result = np.append(final_result, dot_result.toarray())
#one_hot_example = []
#for value in one_hot_data_result:
#    one_hot_example.extend(value)
print len(final_result)
print final_result
#print one_hot_example


#key_list = ['kt_hash','begid','city_code','doc_category','doc_original','interestbizs','age','pic_num','province_code','sex','subcategories','usercategories','video_num','vulgar','tag_list','ttseg_hash']

#key_list = ['province_code']
#key_label_dict = defaultdict(lambda: '[]')




