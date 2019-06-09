
# coding: utf-8

# In[1]:


import numpy as np

import tensorflow as tf

import pandas as pd

from sklearn.metrics import f1_score 

from keras.preprocessing import sequence
from tensorflow.python.keras.utils import to_categorical
from tensorflow.contrib.layers.python.layers import batch_norm


# In[2]:


def read_data(url1, url2, url3, url4):
    train_list = np.load(url1)
    valid_list = np.load(url2)
    test_list = np.load(url3)
    embedding_weights = np.load(url4)
    return train_list, valid_list, test_list, embedding_weights


# In[3]:


train_list1, valid_list1, test_list1, embedding_weights = read_data(
                    r'C:\Users\popzq\Desktop\data\train_list1.npy',
                    r'C:\Users\popzq\Desktop\data\valid_list1.npy',
                    r'C:\Users\popzq\Desktop\data\test_list1.npy',
                    r'C:\Users\popzq\Desktop\data\embedding_weights.npy'
                   )


# In[4]:


d_train = pd.read_csv(r'C:\Users\popzq\Desktop\data\trainingset.csv',engine='python', encoding="utf_8")# 训练数据集
d_valid = pd.read_csv(r'C:\Users\popzq\Desktop\data\validationset.csv',engine='python', encoding="utf_8")# 验证数据集
d_test = pd.read_csv(r'C:\Users\popzq\Desktop\data\testset.csv',engine='python', encoding="utf_8")# 测试数据集


# In[5]:


vocab_dim = 300
maxlen = 100
n_iteration = 1
n_exposures = 10
window_size = 7
batch_size = 64
n_epoch = 20
input_length = 100
num_classes = 4
penalty1 = 0.01


# In[6]:


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    """x 是 array，y是list"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = []
    for i in indices:
        y_shuffle.append(y[i])
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


# In[7]:


sequence_length = 100 #每个句子都是100维的向量； 
num_classes = 4 # 每个columns都是4分类。
hidden_dim = 256 # 全连接层的神经元
keep_prob = 0.1
config = tf.ConfigProto(allow_soft_placement=True)
batch_size = 64
n_input = 128
att_dim = 200
l2_reg = 0.0001

lr0 = 0.0001
lr_decay = 0.99
lr_step = 500

n_epochs = 2


# In[13]:


def train_evl_pred(col, d_train, d_valid, d_test, 
                   train_list1, valid_list1, test_list1,
                   sequence_length, num_classes, 
                   hidden_dim, keep_prob, config, batch_size, 
                   att_dim, l2_reg, lr0, lr_decay, lr_step, n_epochs):
    y_train_total = np.array(d_train[col])
    y_valid_total = np.array(d_valid[col])
    ## 因为标签处于[-2,1]，是真实标签，因此需要加上2，得到[0,3];预测的结果已经-2，因此直接使用。
    y_train_number = y_train_total + 2
    y_valid_number = y_valid_total + 2
    f11 = [] #收集每个epoch的分数
    # 在开头加上这一句话，才能够重复运行。
    tf.reset_default_graph()

    with tf.name_scope('inputs'):   
        # 注意，这里input_y使用的值是[-2,-1,0,1]，而不是one_hot编码
        # 因此，后面在评估准确性的时候，使用的是
        input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x") 
        input_y = tf.placeholder(tf.int32, [None], name='input_y')      

    with tf.device('/gpu:0'), tf.name_scope('embedding'):
        """词向量映射"""
        W = tf.Variable(embedding_weights, name='W')
        embedded_chars = tf.nn.embedding_lookup(W, input_x)
        print('this is embedded_chars shape, ', embedded_chars.get_shape())

    with tf.name_scope('bilstm'):    
        lstm_cell_fw = tf.contrib.rnn.LSTMCell(hidden_dim)
        lstm_cell_bw = tf.contrib.rnn.LSTMCell(hidden_dim)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, inputs=embedded_chars, dtype=tf.float64)
        output = tf.concat((outputs[0], outputs[1]), axis=2) #注意这里是为了attention，把整个序列拼接上，而不是最后一个状态
        output = batch_norm(output)
        print('this is shape of bilstm, ', output.get_shape())   

    with tf.name_scope('attention1'):
        word_arr_W = tf.get_variable(shape=[att_dim, 1], name='attention_matrix1',dtype=tf.float64) #200是512全连接200，200可以修改
        projection = tf.layers.dense(output, att_dim, tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)) # 200同上,激活函数，先降维【400到200】
    #     print(projection.get_shape()) 
        projection = batch_norm(projection)
        alpha = tf.matmul(tf.reshape(projection, shape=[-1, att_dim]), word_arr_W)
    #     print(alpha.get_shape())
        alpha = tf.reshape(alpha, shape=[-1, sequence_length]) 
    #     print(alpha.get_shape())
        alpha = tf.nn.softmax(alpha)
        attention_result = tf.reduce_sum(output*tf.expand_dims(alpha, 2), axis=1) # 在1时间步上进行平均。
        attention_result = tf.cast(attention_result, tf.float32)
        attention_result = batch_norm(attention_result)
        print('this is shape of attention,', attention_result.get_shape())  

    with tf.name_scope('score1'):    
        # 全连接层，后面接dropout和relu激活
        fc1 = tf.layers.dense(inputs=attention_result, units=128, name='fc1', 
                              reuse=tf.AUTO_REUSE)
        fc1 = batch_norm(fc1)
        fc1 = tf.contrib.layers.dropout(fc1, keep_prob=keep_prob) 
        fc1 = tf.nn.relu(fc1)
        fc1 = batch_norm(fc1)
        logits = tf.layers.dense(fc1, num_classes, name='fc2', 
                                 reuse=tf.AUTO_REUSE)
        logits = batch_norm(logits)
        y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1) - 2

        print('this is shape of logits, ', logits.get_shape())
        print('this is shape of classes, ', y_pred_cls.get_shape())    

    with tf.name_scope('optimize'):
        global_step = tf.Variable(0)
        lr = tf.train.exponential_decay(lr0,
                                      global_step,
                                      decay_steps=lr_step,
                                      decay_rate=lr_decay,
                                      staircase=True)
        training_var = tf.trainable_variables()#得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
        regularization_cost = 0.001 * tf.reduce_sum([ tf.nn.l2_loss(tf.cast(v, tf.float32)) 
                                                     for v in training_var ])
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
        loss = tf.reduce_mean(xentropy) + regularization_cost
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

    # 准确性
    with tf.name_scope('accuracy'):
        logits2 = tf.cast(logits, dtype=tf.float32)
        correct_predictions = tf.nn.in_top_k(logits2, input_y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))


    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()    
        sess.run(init)
        best_f1 = 0
        for epoch in range(n_epochs):                
            batch_train = batch_iter(train_list1, y_train_number, batch_size) 
            for x_batch, y_batch in batch_train:                                 
                ## y_batch是真实的标签.
                sess.run(train_op, feed_dict={input_x:x_batch, input_y:y_batch})
            ## 注意：y_pre_cls已经-2，因此这里预测得到的结果就是最后的结果。
            train_pre = sess.run(y_pred_cls, feed_dict={input_x: x_batch})               
            valid_pre = sess.run(y_pred_cls, feed_dict={input_x: valid_list1})

            ff_train = f1_score(train_pre, np.array(y_batch) - 2, average='macro')  
            ff_valid = f1_score(valid_pre, y_valid_number - 2, average='macro')                                                
            acc_train = accuracy.eval(feed_dict={input_x:x_batch, input_y:y_batch})         
            acc_valid = accuracy.eval(feed_dict={input_x:valid_list1, input_y:y_valid_number})       
            if ff_valid > best_f1:
                best_f1 = ff_valid
                saver = tf.train.Saver()    
                saver.save(sess, r'C:\Users\popzq\Desktop\data\model_%s'%col)
                print(epoch+1, "—— Train f1:%.3f;"%ff_train, 
                      " Valid f1:%.3f;"%ff_valid,
                         "Train accuracy:%.3f;"%acc_train, "Valid accuracy:%.3f."%acc_valid)
        saver = tf.train.import_meta_graph(
            r'C:\Users\popzq\Desktop\data\model_%s.meta'%col)
        saver.restore(sess, tf.train.latest_checkpoint("./"))
        y_pre = sess.run(y_pred_cls, feed_dict={input_x:test_list1})
    return y_pre


# In[14]:


# y_pre = train_evl_pred('location_distance_from_business_district', 
#                        d_train, d_valid, d_test, 
#                        train_list1, valid_list1, test_list1,
#                        sequence_length, num_classes, 
#                        hidden_dim, keep_prob, config, batch_size, 
#                        att_dim, l2_reg, lr0, lr_decay, lr_step, 2)


# In[15]:


def master(d_train, d_valid, d_test, train_list1, valid_list1, test_list1,
           sequence_length, num_classes, hidden_dim, keep_prob, config, 
           batch_size, att_dim, l2_reg, lr0, lr_decay, lr_step, n_epochs):
    y_pre_total = []
    for col in list(d_train.columns[3:]):
        print("This is %s."%col)
        y_pre = train_evl_pred(col, d_train, d_valid, d_test, 
                               train_list1, valid_list1, test_list1,
                               sequence_length, num_classes, 
                               hidden_dim, keep_prob, config, batch_size, 
                               att_dim, l2_reg, lr0, lr_decay, lr_step, n_epochs)
        print(np.unique(y_pre))            
        y_pre_total.append(y_pre)
        print("\n")
        print('————————————————————————————————')
    return y_pre_total


# In[16]:


pre_result = master(d_train, d_valid, d_test, train_list1, valid_list1, test_list1,
           sequence_length, num_classes, hidden_dim, keep_prob, config, 
           batch_size, att_dim, l2_reg, lr0, lr_decay, lr_step, 50)


# In[17]:


pre_result


# In[20]:


d_test1 = d_test.copy()
for i in range(20):
    d_test1.iloc[:,i+3]=pre_result[i]


# In[22]:


d_test1.to_excel(r'C:\Users\popzq\Desktop\data\d_test.xlsx')


# ### CNN

# In[8]:


conv1_fmaps = 16
conv1_ksize = 3
conv1_strides = 1
conv1_pad = "SAME"

conv2_fmaps = 32
conv2_ksize = 3
conv2_strides = 1
conv2_pad = "SAME"

pool1_ksize = [1,4,4,1]
pool1_strides = [1,4,4,1]
pool1_padding="VALID"

pool2_ksize = [1,4,4,1]
pool2_strides = [1,4,4,1]
pool2_padding="VALID"


# In[30]:


def cnn_train_evl_pred(col, d_train, d_valid, d_test, train_list1, valid_list1, test_list1,
                       sequence_length, num_classes, hidden_dim, keep_prob, config, batch_size, 
                       att_dim, l2_reg, lr0, lr_decay, lr_step, n_epochs):
    y_train_total = np.array(d_train[col])
    y_valid_total = np.array(d_valid[col])
    ## 因为标签处于[-2,1]，是真实标签，因此需要加上2，得到[0,3];预测的结果已经-2，因此直接使用。
    y_train_number = y_train_total + 2
    y_valid_number = y_valid_total + 2
    f11 = [] #收集每个epoch的分数
    # 在开头加上这一句话，才能够重复运行。
    tf.reset_default_graph()

    with tf.name_scope('inputs'):   
        # 注意，这里input_y使用的值是[-2,-1,0,1]，而不是one_hot编码
        # 因此，后面在评估准确性的时候，使用的是
        input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x") 
        input_y = tf.placeholder(tf.int32, [None], name='input_y')      

    with tf.device('/gpu:0'), tf.name_scope('embedding'):
        """词向量映射"""
        W = tf.Variable(embedding_weights, name='W')
        embedded_chars = tf.nn.embedding_lookup(W, input_x)
        embedded_chars_expand = tf.expand_dims(embedded_chars, -1)
        print('this is embedded_chars shape, ', embedded_chars_expand.get_shape())

    with tf.name_scope('cnn'):    
        conv1 = tf.layers.conv2d(embedded_chars_expand, 
                                filters = conv1_fmaps, 
                                strides = conv1_strides,
                                padding = conv1_pad,
                                name = 'conv1',
                                kernel_size = conv1_ksize,                             
                                activation = tf.nn.relu,
                                reuse = tf.AUTO_REUSE)
        pool1 = tf.nn.relu(tf.nn.max_pool(conv1,
                               ksize = pool1_ksize,
                               strides = pool1_strides,
                               padding = pool1_padding))   
        conv2 = tf.layers.conv2d(pool1, 
                                filters = conv2_fmaps, 
                                strides = conv2_strides,
                                padding = conv2_pad,
                                name = 'conv2',
                                kernel_size = conv2_ksize,                             
                                activation = tf.nn.relu,
                                reuse = tf.AUTO_REUSE)
        pool2 = tf.nn.relu(tf.nn.max_pool(conv2,
                               ksize = pool2_ksize,
                               strides = pool2_strides,
                               padding = pool2_padding))

        print(pool2.get_shape())
        pool_flat = tf.reshape(pool2, shape=[-1, 6 * 18 * 32])
        print(pool_flat.get_shape())            

    with tf.name_scope('score1'):    
        # 全连接层，后面接dropout和relu激活
        fc1 = tf.layers.dense(inputs=pool_flat, units=128, name='fc1', 
                              reuse=tf.AUTO_REUSE)
        fc1 = tf.cast(fc1, tf.float32)
        fc1 = batch_norm(fc1)
        fc1 = tf.contrib.layers.dropout(fc1, keep_prob=keep_prob) 
        fc1 = tf.nn.relu(fc1)
        fc1 = batch_norm(fc1)
        logits = tf.layers.dense(fc1, num_classes, name='fc2', reuse=tf.AUTO_REUSE)
        logits = batch_norm(logits)
        y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1) - 2
        print('this is shape of logits, ', logits.get_shape())
        print('this is shape of classes, ', y_pred_cls.get_shape())    

    with tf.name_scope('optimize'):
        global_step = tf.Variable(0)
        lr = tf.train.exponential_decay(lr0,
                                      global_step,
                                      decay_steps=lr_step,
                                      decay_rate=lr_decay,
                                      staircase=True)
        training_var = tf.trainable_variables()#得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
        regularization_cost = 0.001 * tf.reduce_sum([ tf.nn.l2_loss(tf.cast(v, tf.float32)) 
                                                     for v in training_var ])
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
        loss = tf.reduce_mean(xentropy) + regularization_cost
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

    # 准确性
    with tf.name_scope('accuracy'):
        logits2 = tf.cast(logits, dtype=tf.float32)
        correct_predictions = tf.nn.in_top_k(logits2, input_y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))


    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()    
        sess.run(init)
        best_valid_f1 = 0
        best_train_f1 = 0
        for epoch in range(n_epochs):                
            batch_train = batch_iter(train_list1, y_train_number, batch_size) 
            for x_batch, y_batch in batch_train:                                 
                ## y_batch是真实的标签.
                sess.run(train_op, feed_dict={input_x:x_batch, input_y:y_batch})
            ## 注意：y_pre_cls已经-2，因此这里预测得到的结果就是最后的结果。
            train_pre = sess.run(y_pred_cls, feed_dict={input_x: x_batch})               
            valid_pre1 = sess.run(y_pred_cls, feed_dict={input_x: valid_list1[:1000]})
            valid_pre2 = sess.run(y_pred_cls, feed_dict={input_x: valid_list1[1000:]})

            ff_train = f1_score(train_pre, np.array(y_batch) - 2, average='macro')  
            ff_valid1 = f1_score(valid_pre1, y_valid_number[:1000] - 2, average='macro')                                                
            ff_valid2 = f1_score(valid_pre2, y_valid_number[1000:] - 2, average='macro')                                                
            
            acc_train = accuracy.eval(feed_dict={input_x:x_batch, input_y:y_batch})         
            acc_valid1 = accuracy.eval(feed_dict={input_x:valid_list1[:1000], input_y:y_valid_number[:1000]})       
            acc_valid2 = accuracy.eval(feed_dict={input_x:valid_list1[1000:], input_y:y_valid_number[1000:]})                   
            max_valid_ff = np.max([ff_valid1, ff_valid2])
            if (max_valid_ff > best_valid_f1) and (ff_train > 0.6):
                best_valid_f1 = max_valid_ff
#                 saver = tf.train.Saver()    
#                 saver.save(sess, r'C:\Users\popzq\Desktop\data\model_%s'%col)
                print(epoch+1, "—— Train f1:%.3f;"%ff_train, 
                      " Valid f1:%.3f;"%best_valid_f1,
                         "Train accuracy:%.3f;"%acc_train, "Valid accuracy:%.3f."%np.max([acc_valid1, acc_valid2]))
#             saver = tf.train.import_meta_graph(r'C:\Users\popzq\Desktop\data\cnn_model_%s.meta'%col)
#             saver.restore(sess, tf.train.latest_checkpoint("./"))
#             y_pre = sess.run(y_pred_cls, feed_dict={input_x:test_list1})
#     return y_pre


# In[31]:


def master_cnn(d_train, d_valid, d_test, train_list1, valid_list1, test_list1,
           sequence_length, num_classes, hidden_dim, keep_prob, config, 
           batch_size, att_dim, l2_reg, lr0, lr_decay, lr_step, n_epochs):
    y_pre_total = []
    for col in list(d_train.columns[3:]):
        print("This is %s."%col)
        y_pre = cnn_train_evl_pred(col, d_train, d_valid, d_test, 
                               train_list1, valid_list1, test_list1,
                               sequence_length, num_classes, 
                               hidden_dim, keep_prob, config, batch_size, 
                               att_dim, l2_reg, lr0, lr_decay, lr_step, n_epochs)
        print(np.unique(y_pre))            
        y_pre_total.append(y_pre)
        print("\n")
        print('————————————————————————————————')
    return y_pre_total


# In[ ]:


pre_result = master_cnn(d_train, d_valid, d_test, train_list1, valid_list1, test_list1,
           sequence_length, num_classes, hidden_dim, keep_prob, config, 
           batch_size, att_dim, l2_reg, lr0, lr_decay, lr_step, 100)

