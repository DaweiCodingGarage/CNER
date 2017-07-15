import re


# Design features in this function
def word2features_1(sent, i,word_set_suffix,word_set_prefix):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word':word,
        'word.ispunc()': 1 if re.match('^[a-zA-Z0-9_]*$',word) else 0,
        'word.isdigit()': word.isdigit(),
        'word.isalpha()': word.isalpha(),

    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({

            '-1:word':word1,
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.isalpha()': word1.isalpha(),
            '-1:word.ispunc()': 1 if re.match('^[a-zA-Z0-9_]*$',word1) else 0,
            '-1:0.word.suffix':   word1+word in word_set_suffix,
            '-1:0.word.prefix':   word1+word in word_set_prefix,   
            '-1:0': word1+word,              
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word2 = sent[i+1][0]
        #postag2 = sent[i+1][1]
        #head2 = sent[i+1][2]
        features.update({
            '+1:word':word2,
            '+1:word.isdigit()': word2.isdigit(),
            '+1:word.isalpha()': word2.isalpha(),
            '+1:word.ispunc()': 1 if re.match('^[a-zA-Z0-9_]*$',word2) else 0,
            '+1:0': word+word2,
            '+1:0.word.suffix':   word+word2 in word_set_suffix,
            '+1:0.word.prefix':   word+word2 in word_set_prefix,
           
        })
    else:
        features['EOS'] = True
        
    if i > 1:
        word3 = sent[i-2][0]
        features.update({
             '-2:word': word3,
            '-2:-1:word': word3+word1,
            '-2:-1:0_word': word3+word1+word,
           '-2:-1.word.suffix':  word3+word1 in word_set_suffix,
            '-2:-1.word.prefix':   word3+word1 in word_set_prefix,
            '-2:-1:0.word.suffix':  word3+word1+word in word_set_suffix,
            '-2:-1:0.word.prefix':   word3+word1+word in word_set_prefix,          
        })
        
    if i < len(sent)-2:
        word4 = sent[i+2][0]
        features.update({
             '+2:word': word4,
            '+2:+1:word': word2+word4,
            '+2:+1.word.suffix': word2+word4 in word_set_suffix,
            '+2:+1.word.prefix':   word2+word4 in word_set_prefix,
            '+2:+1:0_word': word+word2+word4,
            '+2:+1:0.word.suffix':  word+word2+word4 in word_set_suffix,
            '+2:+1:0.word.prefix':   word+word2+word4 in word_set_prefix,
           
        })

    return features
