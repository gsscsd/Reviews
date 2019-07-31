### 神经语言模型

```python
# coding=utf-8
'''
Created on 2018年9月15日

@author: admin
'''

from gensim import corpora, models, similarities
import numpy as np
if __name__ == '__main__':
    text = [["我","今天","打","篮球"],
            ["我","今天","打","足球"],
            ["我","今天","打","羽毛球"],
            ["我","今天","打","网球"],
            ["我","今天","打","排球"],
            ["我","今天","打","气球"],
            ["我","今天","打","游戏"],
            ["我","今天","打","冰球"],
            ["我","今天","打","人"],
            ["我","今天","打","台球"],
            ["我","今天","打","桌球"],
            ["我","今天","打","水"],
            ["我","今天","打","篮球"],
            ["我","今天","打","足球"],
            ["我","今天","打","羽毛球"],
            ["我","今天","打","网球"],
            ["我","今天","打","排球"],
            ["我","今天","打","气球"],
            ]
    #使用gensim生成词典
    dictionary = corpora.Dictionary(text,prune_at=2000000)
    #打印词典中的词
    for key in dictionary.iterkeys():
        print(key,dictionary.get(key),dictionary.dfs[key])
    #保存词典
    dictionary.save_as_text('word_dict.dict',  sort_by_word=True)
    #加载词典
    dictionary = dictionary.load_from_text('word_dict.dict')
    
    #词语个数
    word_num = len(dictionary.keys())
    #使用多少编文章生成每个batch数据
    sentence_batch_size = 1
    #滑动窗口
    window = 3
    #生成CBOW数据
    def data_generator(): #训练数据生成器
        while True:
            x,y = [],[]
            _ = 0
            for sentence in text:
	            #在两端插入空字符，这里使用word_num这个值来代替
                sentence = [word_num]*window + [dictionary.token2id[w] for w in sentence if w in dictionary.token2id] + [word_num]*window
                for i in range(window, len(sentence)-window):
                    x.append(sentence[i-window:i]+sentence[i+1:i+1+window])
                    #因为使用的loss函数为sparse_categorical_crossentropy，所以不用one-hot
                    y.append([sentence[i]])
                _ += 1
                if _ == sentence_batch_size:
                    x,y = np.array(x),np.array(y)
                    print("输入的数据 :",x.shape)
                    print("对应的标签 :",y.shape)
                    yield x,y
                    x,y = [],[]
                    _ = 0
       
    from keras.models import Sequential
    from keras.layers import Dense, Activation,Embedding,Reshape
    model = Sequential()
    model.add(Embedding(word_num+1, 200, input_length=6))
    model.add(Dense(1024, activation='relu'))
    model.add(Flatten())
    model.add(Dense(word_num+1, activation='softmax'))
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    model.fit_generator(data_generator(),steps_per_epoch =np.ceil(dictionary.num_docs/sentence_batch_size),epochs=1000,max_queue_size=1,workers=1)
    #保存模型
    model.save_weights("DNNword-vec.h5")
    #加载模型
    model.load_weights("DNNword-vec.h5",by_name=True)
    
    #获取embeding的权重，也就是词向量
    embeddings = model.get_weights()[0]
    #向量标准化
    normalized_embeddings = embeddings / (embeddings**2).sum(axis=1).reshape((-1,1))**0.5
    dictionary.id2token = {j:i for i,j in dictionary.token2id.items()}
    #获取前面最相似的10个词语
    def most_similar(w,dictionary):
        v = normalized_embeddings[dictionary.token2id[w]]
        #向量标准化之后分母就是1，所以直接相乘就好
        sims = np.dot(normalized_embeddings, v)
        sort = sims.argsort()[::-1]
        sort = sort[sort > 0]
        #如果是占位符则不输出
        return [(dictionary.id2token[i],sims[i]) for i in sort[:10] if i in dictionary.id2token]
    
    for sim in most_similar(u'网球',dictionary):
        print(sim[0],sim[1])
# 网球 1.0000001
# 羽毛球 0.11263792
# 桌球 0.07463527
# 篮球 0.066648
# 足球 0.06379064
# 台球 0.046809338
# 排球 0.04252596
# 我 0.04014937
# 人 0.028555304
# 水 0.007580313
```

### word2vec+负采样

```python
# coding=utf-8
'''
Created on 2018年9月15日

@author: admin
'''

from gensim import corpora, models, similarities
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
class NegativeLayer(Layer):
    def __init__(self, nb_negative,M,M_num, **kwargs):


        self.nb_negative = nb_negative
        self.M = M
        self.M_num = M_num
        super(NegativeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NegativeLayer, self).build(input_shape)


    def call(self, x, mask=None):

        batch = 0
        if str(x.shape[0]).isdigit() == False:
            batch = 4
        else:
            batch = x.shape[0]
        #负采样
        final_output = np.array([[M[i] for i in j]for j in np.random.randint(0, self.M_num+1, size=(batch, self.nb_negative))])
        #变成tensor格式
        final_output = K.tensorflow_backend._to_tensor(final_output,dtype=np.int32)
        return final_output
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_negative)
if __name__ == '__main__':
    text = [["我","今天","打","篮球"],
            ["我","今天","打","足球"],
            ["我","今天","打","羽毛球"],
            ["我","今天","打","网球"],
            ["我","今天","打","排球"],
            ["我","今天","打","气球"],
            ["我","今天","打","游戏"],
            ["我","今天","打","冰球"],
            ["我","今天","打","人"],
            ["我","今天","打","台球"],
            ["我","今天","打","桌球"],
            ["我","今天","打","水"],
            ["我","今天","打","篮球"],
            ["我","今天","打","足球"],
            ["我","今天","打","羽毛球"],
            ["我","今天","打","网球"],
            ["我","今天","打","排球"],
            ["我","今天","打","气球"],
            ]
    #使用gensim生成词典
    dictionary = corpora.Dictionary(text,prune_at=2000000)
    
    #打印词典中的词
    for key in dictionary.iterkeys():
        print(key,dictionary.get(key),dictionary.dfs[key])
    #保存词典
    dictionary.save_as_text('word_dict.dict',  sort_by_word=True)
    #加载词典
    dictionary = dictionary.load_from_text('word_dict.dict')
    
    L = {}
    #计算出词出现的总数，dictionary.dfs{单词id，在多少文档中出现}
    allword_num = np.sum(list(dictionary.dfs.values()))
    print(allword_num)
    #72
    
    #构造负采样dict
    #进行归一化,然后按照0-1排列,然后再使用M个均等值来评分0-1，方便对应词的id
    sum = 0
    M = {}
    M_num = 1000
    for id,num in dictionary.dfs.items():
        #向上取整
        left = int(np.ceil(sum/(1/M_num)))
        sum = sum + num/allword_num
        L[id] = sum
        #向下取整
        right = int(sum/(1/M_num))
        print(id,left,right)
#         11 0 13
#         0 14 263
#         10 264 277
#         12 278 291
#         1 292 541
#         2 542 791
#         7 792 819
#         13 820 833
#         8 834 861
#         14 862 875
#         9 875 888
#         3 889 916
#         6 917 944
#         5 945 972
#         4 973 1000
        for i in range(left,right+1):
            M[i] = id
    print(L)
    #{11: 0.013888888888888888, 0: 0.25, 10: 0.013888888888888888, 12: 0.013888888888888888, 1: 0.25, 2: 0.25, 7: 0.027777777777777776, 13: 0.013888888888888888, 8: 0.027777777777777776, 14: 0.013888888888888888, 9: 0.013888888888888888, 3: 0.027777777777777776, 6: 0.027777777777777776, 5: 0.027777777777777776, 4: 0.027777777777777776}
        
    #词语个数
    word_num = len(dictionary.keys())
    #使用多少编文章生成每个batch数据
    sentence_batch_size = 1
    #滑动窗口
    window = 3
    def data_generator(): #训练数据生成器
        while True:
            x,y = [],[]
            _ = 0
            for sentence in text:
                #使用word_num的值作为padding
                sentence = [word_num]*window + [dictionary.token2id[w] for w in sentence if w in dictionary.token2id] + [word_num]*window
                for i in range(window, len(sentence)-window):
                    x.append(sentence[i-window:i]+sentence[i+1:i+1+window])
                    #因为使用的loss函数为sparse_categorical_crossentropy，所以不用one-hot
                    y.append([sentence[i]])
                _ += 1
                if _ == sentence_batch_size:
                    x,y = np.array(x),np.array(y)
                    #因为正例为输出层第一个神经元，所以这里都使用0标签，也是因为loss函数为sparse_categorical_crossentropy
                    z = np.zeros((len(x), 1))
                    print("输入的数据 :",x.shape)
                    print("对应的标签 :",y.shape)
                    print("对应的标签 2:",z.shape)
                    yield [x,y],z
                    x,y = [],[]
                    _ = 0
        
    from keras.models import Sequential
    from keras.layers import Dense, Activation,Embedding,Reshape,Flatten,Input,Embedding,Lambda

    from keras.models import Model
    #词向量维度
    word_size = 100
    #负样本个数
    nb_negative = 16
    
    input_words = Input(shape=(window*2,), dtype='int32')
    input_vecs = Embedding(word_num+1, word_size, name='word2vec')(input_words)
    input_vecs_sum = Lambda(lambda x: K.sum(x, axis=1))(input_vecs) #CBOW模型，直接将上下文词向量求和
    
    #构造随机负样本，与目标组成抽样
    target_word = Input(shape=(1,), dtype='int32')
    negatives = NegativeLayer(16,M,M_num)(target_word)
    samples = Lambda(lambda x: K.concatenate(x))([target_word,negatives]) #构造抽样，负样本随机抽。负样本也可能抽到正样本，但概率小。
    
    #使用Embedding层代替dense主要原因是只更新正例和负例相对应的输出层神经元的权重，这样可以大量减少内存占用和计算量
    softmax_weights = Embedding(word_num+1, word_size, name='W')(samples)
    softmax_biases = Embedding(word_num+1, 1, name='b')(samples)
    softmax = Lambda(lambda x: 
                        K.softmax((K.batch_dot(x[0], K.expand_dims(x[1],2))+x[2])[:,:,0])
                    )([softmax_weights,input_vecs_sum,softmax_biases]) #用Embedding层存参数，用K后端实现矩阵乘法，以此复现Dense层的功能
    
    #留意到，我们构造抽样时，把目标放在了第一位，也就是说，softmax的目标id总是0，这可以从data_generator中的z变量的写法可以看出
    
    model = Model(inputs=[input_words,target_word], outputs=softmax)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    

     
    model.summary()
    model.fit_generator(data_generator(),steps_per_epoch =np.ceil(dictionary.num_docs/sentence_batch_size),epochs=100,max_queue_size=1,workers=1)
#     #保存模型
    model.save_weights("DNNword-vec2.h5")
#     #加载模型
    model.load_weights("DNNword-vec2.h5",by_name=True)
#     
    #获取embeding的权重，也就是词向量
    embeddings = model.get_weights()[0]
    #向量标准化
    normalized_embeddings = embeddings / (embeddings**2).sum(axis=1).reshape((-1,1))**0.5
    dictionary.id2token = {j:i for i,j in dictionary.token2id.items()}
    #获取前面最相似的15个词语
    def most_similar(w,dictionary):
        v = normalized_embeddings[dictionary.token2id[w]]
        #向量标准化之后分母就是1，所以直接相乘就好
        sims = np.dot(normalized_embeddings, v)
        sort = sims.argsort()[::-1]
        sort = sort[sort > 0]
        return [(dictionary.id2token[i],sims[i]) for i in sort[:15] if i in dictionary.id2token]
     
    for sim in most_similar(u'网球',dictionary):
        print(sim[0],sim[1])
# 网球 0.99999994
# 羽毛球 0.9787248
# 篮球 0.978495
# 排球 0.9773369
# 人 0.9761201
# 水 0.9760275
# 气球 0.9753146
# 桌球 0.9731983
# 冰球 0.97278094
# 游戏 0.9711289
# 足球 0.9660615
# 台球 0.96072686
# 我 -0.3409065
# 打 -0.42166257
```

