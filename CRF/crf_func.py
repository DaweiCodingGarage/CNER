import re
from itertools import chain
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
from utils import input_data_transform,write_conll,test_ner
from features import word2features_1



def sent2features(sent,word_set_suffix,word_set_prefix):
    return [word2features_1(sent, i,word_set_suffix,word_set_prefix) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token in sent]


def crf_tune_hyperparam(data,index, label,word_set_suffix,word_set_prefix,max_iterations =500):
    train_data = [data[i] for i in index]
    X = [sent2features(s,word_set_suffix,word_set_prefix) for s in train_data]
    y = [sent2labels(s) for s in train_data]
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=max_iterations,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }
    label.remove("O")
    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted',labels=label)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=8,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(X, y)
    return rs.best_params['c1'],rs.best_params['c2']

def train_test_split(data, train_index, test_index,word_set_suffix,word_set_prefix):
    X = [sent2features(s,word_set_suffix,word_set_prefix) for s in data]
    y = [sent2labels(s) for s in data]

    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    return X_train,y_train,X_test,y_test


def CRF_run(X_train,y_train,X_test, word_set_suffix,word_set_prefix,best_c1=0.1,best_c2=0.1,max_iterations=500):
    
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=best_c1,
        c2=best_c2,
        max_iterations=max_iterations,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)

    return crf,y_pred

def CRF_eval(data,test_index,y_pred,path,self_eval):
    test_char = [data[i] for i in test_index]
    if self_eval:
        datawpred = [[[data[0],data[-1]]+[pred] for data, pred in zip(test_char[j],y_pred[j])] for j in range(len(y_pred))]
    else:
        datawpred = [[[data,pred] for data, pred in zip(test_char[j],y_pred[j])] for j in range(len(y_pred))]
    with open(path+"pred{}.conll".format(self_eval!=True),'w',encoding='utf-8') as f:
        write_conll(f, input_data_transform(datawpred))
    if self_eval:
        test_ner(path)

    
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
