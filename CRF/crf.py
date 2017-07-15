# -*- coding: gbk -*- 
from utils import load_data,load_med_set,word_set_prefix_and_suffix,load_test_data,output_data_w_required_format
from sklearn.cross_validation import KFold
from collections import Counter
from crf_func import CRF_run,crf_tune_hyperparam,print_state_features,train_test_split,CRF_eval,sent2features




if __name__ == "__main__":
    ## default parameters
    path = ".\\"
    data_path = "..\\"
    task_data_dir, eval_data_dir = "task2data","task2test"
    dict_name = u"字典全.txt"
    label = [u'检查和检验', u'治疗', u'疾病和诊断', u'症状和体征', u'身体部位']
    tag = ['B-0','I-0','B-1','I-1','B-2','I-2','B-3','I-3','B-4','I-4','O']
    rnd_seed = 2
    best_c1, best_c2 = 0.089, 0.004
    test = True
    tuned_params = False

    #load data
    label_data_list = load_data(data_path,task_data_dir,label)
    # load medical dictionary
    med_set = load_med_set(dict_name)
    # extracted common 2gram, 3gram, 4gram, prefix, and suffix
    word_set_suffix, word_set_prefix = word_set_prefix_and_suffix(med_set)
    # split data randomly into 9:1
    
    # tune parameter for c1 and c2
    if tuned_params:
        
        best_c1, best_c2 = crf_tune_hyperparam(label_data_list,train_index,\
                                               tag,word_set_suffix, word_set_prefix)
        print("best c1 is {} and best c2 is {} after tuning hyperparameter".format(best_c1, best_c2))

    if test: ## evalute test data with no labels for competition submission
        ## load data from test data folder
        test_data,file_name = load_test_data(data_path,eval_data_dir)
        
        X,y,_,_ = train_test_split(label_data_list,range(len(label_data_list)),[],\
                                   word_set_suffix, word_set_prefix)
        
        X_test = [sent2features(s,word_set_suffix,word_set_prefix) for s in test_data]
        

        crf_all, y_pred_test = CRF_run(X,y,X_test, \
                                      word_set_suffix, word_set_prefix,best_c1,best_c2)
        CRF_eval(test_data,range(len(test_data)),y_pred_test,path,False)

        print("Top positive:")
        print_state_features(Counter(crf_all.state_features_).most_common(100))

        print("\nTop negative:")
        print_state_features(Counter(crf_all.state_features_).most_common()[-100:][::-1])


        #output in the required format
        output_data_w_required_format(path+"predTrue.conll",file_name,label,path+"result.csv")
    else:
        kf = KFold(len(label_data_list), n_folds=10,shuffle=True,random_state=rnd_seed)
        
        train_index, test_index = list(kf)[0]
        # split into train, test data
        X_train,y_train,X_test,y_test = train_test_split(label_data_list,train_index,\
                                                        test_index,word_set_suffix, word_set_prefix)
        # run CRF
        crf, y_pred = CRF_run(X_train,y_train,X_test, \
                                                word_set_suffix, word_set_prefix,best_c1,best_c2)
        # evaluate CRF
        CRF_eval(label_data_list,test_index,y_pred,path,True)
        ## print most significant features
        print("Top positive:")
        print_state_features(Counter(crf.state_features_).most_common(100))

        print("\nTop negative:")
        print_state_features(Counter(crf.state_features_).most_common()[-100:][::-1])
        
    
    
