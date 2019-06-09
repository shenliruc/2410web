
# coding: utf-8

# In[21]:


import re
import numpy as np
import pandas as pd
import tensorflow as tf
import jieba


# In[17]:


vocab_dim = 300
maxlen = 100


# In[2]:


d_train = pd.read_csv(r'C:\Users\popzq\Desktop\data\trainingset.csv',engine='python', encoding="utf_8")# 训练数据集
d_valid = pd.read_csv(r'C:\Users\popzq\Desktop\data\validationset.csv',engine='python', encoding="utf_8")# 验证数据集
d_test = pd.read_csv(r'C:\Users\popzq\Desktop\data\testset.csv',engine='python', encoding="utf_8")# 测试数据集


# In[3]:


def read_vectors(path):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim


# In[4]:


#导入停止词
def get_stop_word(url):
    stop_word = []
    with open(url, encoding='UTF-8') as fp:
        for i in fp:
            zh = re.compile("[^\u4e00-\u9fa5]") #匹配不是中文的字符
            i = zh.sub('', i)
            stop_word.append(i)        
    return stop_word


# In[5]:


def segmentWord(cont, stop_word, types):
    if types == 'pre_train':
        c = [] 
        for i in cont:
            zh = re.compile("[^\u4e00-\u9fa5]") #匹配不是中文的字符
            string1 = zh.sub('', i)             #将string1中匹配到的字符替换成空字符
            a = list(jieba.cut(string1))
            #删去停止词
            a1 = [item for item in a if item not in stop_word]
            if len(a1) >= 10:
                b = []
                b.append(a1)
                c.extend(b)
    else:
        c = [] 
        for i in cont:
            zh = re.compile("[^\u4e00-\u9fa5]") #匹配不是中文的字符
            string1 = zh.sub('', i)             #将string1中匹配到的字符替换成空字符
            a = list(jieba.cut(string1))
            #删去停止词
            a1 = [item for item in a if item not in stop_word]
            b = []
            b.append(a1)
            c.extend(b)        
    return c


# In[7]:


def get_sentence_code(text_list, w2indx):
    # 将每句话对应的编号存在data中。
    data = []
    for sentence in text_list:
        new_txt = []
        for word in sentence: #将一句话中的每个词语先对应编号，放到new_txt中。
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0) #同时注意，字典中没有出现的字作为0放入句子编码中。
        data.append(new_txt)
    return data

def get_model(w2indx, train_list, valid_list, test_list, maxlen): #
    # vectors 是{w:vec}, wi是{w: indx}
    train_list = get_sentence_code(train_list, w2indx)
    valid_list = get_sentence_code(valid_list, w2indx)
    test_list = get_sentence_code(test_list, w2indx)
    
    # 统一长度
    train_list = sequence.pad_sequences(train_list, maxlen=maxlen)
    valid_list = sequence.pad_sequences(valid_list, maxlen=maxlen)
    test_list = sequence.pad_sequences(test_list, maxlen=maxlen)
    return train_list, valid_list, test_list


# In[18]:


def get_embedding(w2indx, w2vec, vocab_dim):
    #这是因为，没有出现的字的编号=0，其他字典中出现的字是从1开始。
    n_symbols = len(w2indx) + 1 
    # 初始化词向量矩阵
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    # 生成词向量矩阵
    for word, index in w2indx.items():
        embedding_weights[index, :] = w2vec[word]
    print(embedding_weights)
    print("————————————————————————————————")
    print(np.shape(embedding_weights))
    return embedding_weights


# In[19]:


def preprocess_data(embd_url, stop_word_url, maxlen, vocab_dim, 
                    d_train, d_valid, d_test):
    vectors, iw, wi, dim = read_vectors(embd_url)
    stop_word_dp = get_stop_word(stop_word_url)
    train_cont = np.array(d_train['content'])
    test_cont = np.array(d_test['content'])
    valid_cont = np.array(d_valid['content'])
    train_list = segmentWord(train_cont, stop_word_dp, 'else')
    valid_list = segmentWord(valid_cont, stop_word_dp, 'else')
    test_list = segmentWord(test_cont, stop_word_dp, 'else')
    train_list1, valid_list1, test_list1 = get_model(wi, train_list, valid_list, test_list, maxlen)
    embedding_weights = get_embedding(wi, vectors, vocab_dim)
    return train_list1, valid_list1, test_list1, embedding_weights


# In[20]:


train_list1, valid_list1, test_list1, embedding_weights = preprocess_data(
                r'C:\Users\popzq\Desktop\data\sgns_sogou_word\sgns.sogou.word',
                r'C:\Users\popzq\Desktop\data\dzdpstopwords.txt',
                maxlen, vocab_dim, d_train, d_valid, d_test)


# In[ ]:


np.save(r'C:\Users\popzq\Desktop\data\train_list1.npy', train_list1)
np.save(r'C:\Users\popzq\Desktop\data\valid_list1.npy', valid_list1)
np.save(r'C:\Users\popzq\Desktop\data\test_list1.npy', test_list1)
np.save(r'C:\Users\popzq\Desktop\data\embedding_weights.npy', embedding_weights)

