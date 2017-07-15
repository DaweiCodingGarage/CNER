from collections import defaultdict

import pandas as pd
import os
import csv
import re


# load ccks data into list of list format
def load_data(path,folder_name,label):
    label2id = {l:ind for ind, l in enumerate(label)}
    label_data_list = []
    feature_dict = defaultdict(dict)
    directory = os.path.join(path,folder_name)
    for root,dirs,files in os.walk(directory):
        for ind, file in enumerate(files):
            if file.endswith(".txt"):
                if ind % 2 == 0:
                    #tree = Trie()
                    with open(root+'\\'+file,'r',encoding='utf-8') as infile:
                        reader = csv.reader(infile,delimiter='\t')
                        tag_seq = []                    
                        for line in reader:
                            if line:
                                #store entity position
                                tag_seq.append((line[1],line[2],label2id[line[3]]))
                                                         
                else:
                    #print(file)
                    data_label = []
                    with open(root+'\\'+file,'r',encoding='utf-8') as infile:
                        reader = csv.reader(infile,delimiter='\n')
                        for line in reader:
                            if line:
                                temp = line[0].replace("\t"," ")
                                data_label += list(temp)
                            
                    #print(question)
                    data_label = [[i,j] for i, j in zip(data_label,['O',]*len(data_label))]
                    #print(len(data_label),len(tag_seq))
                    if len(tag_seq):
                        for start,end,tag in tag_seq:
                            for i in range(int(start),int(end)+1):
                                if i == int(start):

                                    data_label[i][-1] ='B-'+str(tag)
                                else:
                                    data_label[i][-1] ='I-'+str(tag)
                    
                                
                    label_data_list.append(data_label)
    return label_data_list


## transform data into a format that write_coll can write
def input_data_transform(docs):
    input_data = []
    for i in map(lambda x:list(zip(*x)),docs):
        input_data.append([list(j) for j in i])

    return input_data

## write data into conll format
def write_conll(fstream, data):
    """
    Writes to an output stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
    @data a list of examples [(tokens), (labels), (predictions)]. @tokens, @labels, @predictions are lists of string.
    """
    for cols in data:
        for row in zip(*cols):
            fstream.write("\t".join([str(i) for i in row]))
            fstream.write("\n")
        fstream.write("\n")
    
## Entitiy Evaluation
def test_ner(output_path):
    script_file = "./conlleval"
    output_file = output_path+ "predFalse.conll"
    result_file = output_path+ "ner_resultFalse.utf8"
    
    os.system("perl {} < {} > {}".format(script_file, output_file, result_file))



def get_entity_type(tag_list):
    normal_tag_list = ['0','1','2','3','4']## label id name
    max_count = 0
    for tag in normal_tag_list:
        if tag_list.count(tag) > max_count:
            max_count = tag_list.count(tag)
            real_tag = tag
    return real_tag


def generate_result(file_path):
    sents=[]  #keep results
    tempLine=[] #keep temporary result
    for eachLine in open(file_path,'r',encoding='utf8'): 
        if(eachLine!='\n'): 
            colList=eachLine.strip('\n').split('\t') 
            #print(colList)
            tempLine.append([colList[0],colList[1]]) 
        else: 
            sents.append(tempLine[:]) 
            tempLine=[] 
    #print(sents)
    #print(len(sents))

    final_results=[] 
    for sentId in range(len(sents)): 
        sentence_result = []
        entity_word='' 
        tag_list = []
        firstWordId=0 
        while(firstWordId<len(sents[sentId])-1): 
            if(sents[sentId][firstWordId][-1]!='O' ) and ('B-' in sents[sentId][firstWordId][-1]): 
                secondWordId=firstWordId+1 
                tag_list.append(sents[sentId][firstWordId][-1].split('-')[-1])
                entity_word += sents[sentId][firstWordId][0] 
                start_index = firstWordId
                while(secondWordId<len(sents[sentId])):
                    if(sents[sentId][secondWordId][-1]!='O') and ('B-' not in sents[sentId][secondWordId][-1]): 
                    
                        entity_word += sents[sentId][secondWordId][0] 
                        tag_list.append(sents[sentId][secondWordId][-1].split('-')[-1])
                    else: 
                        break 
                    secondWordId+=1
                    firstWordId+=1
                real_tag = get_entity_type(tag_list)
                index = (start_index,secondWordId)
                sentence_result.append({'index':index, 'value':(real_tag, entity_word)})
               
                entity_word='' 
                tag_list = []
            firstWordId+=1 
        final_results.append(sentence_result)
        #print(final_results)
    return final_results


## output the desired evaluation format
def get_str_result(file_path,label):
    tag_mapping_dict = {str(ind):l for ind, l in enumerate(label)}
    ## corresponding entity name
    final_result = generate_result(file_path)
    #final_result = post_process(final_result)
    final_str_result =[]
    for i in range(len(final_result)):
        all_text_str = ''
        all_text_list = []
        for dict_ in final_result[i]:
            text_str = ' '.join([dict_['value'][1],str(dict_['index'][0]),\
                                 str(dict_['index'][1]-1),tag_mapping_dict[dict_['value'][0]]])
            all_text_list.append(text_str[:])
        all_text_str = ';'.join(all_text_list)
        final_str_result.append(all_text_str)
    return final_str_result




def load_med_set(dict_name):
    med_set = set()
    med_dict = pd.read_csv(dict_name,names=['word'])
    for word in med_dict['word'].tolist():
        if word not in med_set:
            med_set.add(word)
    return med_set



def common_suffix(med_set, cutoff):
    suffix_dict = {}
    for i in med_set:
        if len(i)>=4:
            word_list = [i[-2:],i[-3:],i[-4:]]
        elif len(i)>=3:
            word_list = [i[-2:],i[-3:]]
        elif len(i)>=2:
            word_list = [i[-2:]]
        else:
            word_list=[]
        
        for word in word_list:
            if word not in suffix_dict:
                suffix_dict[word] = 1
            else:
                suffix_dict[word] += 1
    return {key:value for key,value in suffix_dict.items() if value>=cutoff}

def common_prefix(med_set,cutoff):
    prefix_dict = {}
    for i in med_set:
        if len(i)>=4:
            word_list = [i[:2],i[:3],i[:4]]
        elif len(i)>=3:
            word_list = [i[:2],i[:3]]
        elif len(i)>=2:
            word_list = [i[:2]]
        else:
            word_list=[]
        
        for word in word_list:
            if word not in prefix_dict:
                prefix_dict[word] = 1
            else:
                prefix_dict[word] += 1
    return {key:value for key,value in prefix_dict.items() if value>=cutoff}

def word_set_prefix_and_suffix(med_set):
    word_set_suffix =set()
    word_set_prefix = set()

    suffix = common_suffix(med_set,1)
    prefix = common_prefix(med_set,1)
    for key, value in suffix.items():    
        word_set_suffix.add(key)

    for key, value in prefix.items():    
        word_set_prefix.add(key)
    return word_set_suffix, word_set_prefix


def load_test_data(path,dir1):
    data_list = []
    file_list = []
    directory = os.path.join(path,dir1)
    for root,dirs,files in os.walk(directory):
        for ind, file in enumerate(files):
            if file.endswith(".txt"):
                file_cat, file_ind = file.split(".")[0].split("-")
                file_list.append(",".join([file_ind,file_cat]))
                    #print(file)
                data_label = []
                with open(root+'\\'+file,'r',encoding='utf-8') as infile:
                    reader = csv.reader(infile,delimiter='\n')
                    for line in reader:
                        if line:
                            temp = line[0].replace("\t"," ")
                            data_label += list(temp)

                
                data_list.append(data_label[:])
    return data_list, file_list

def output_data_w_required_format(pred_file_path,file_list,label,output_name='result.csv'):
    final = get_str_result(pred_file_path,label)
    with open(output_name,'w',encoding='utf-8') as f:
   
        for i in range(len(final)):
            if final[i]:
                f.write(file_list[i]+","+final[i]+";")
                f.write("\n")
            else:
                f.write(file_list[i]+",")
                f.write("\n")
    

