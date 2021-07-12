#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: admin
@file: FAQ.py
@time: 2021/07/08
@desc:
"""
import json


# ### 第一部分：对于训练数据的处理：读取文件和预处理

# - ```文本的读取```： 需要从文本中读取数据，此处需要读取的文件是```dev-v2.0.json```，并把读取的文件存入一个列表里（list）
# - ```文本预处理```： 对于问题本身需要做一些停用词过滤等文本方面的处理
# - ```可视化分析```： 对于给定的样本数据，做一些可视化分析来更好地理解数据


# #### 1.1节： 文本的读取
# 把给定的文本数据读入到```qlist```和```alist```当中，这两个分别是列表，其中```qlist```是问题的列表，```alist```是对应的答案列表
def read_corpus():
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。 在此过程中，不用对字符换做任何的处理（这部分需要在 Part 2.3里处理）
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["答案1", "答案2", "答案3" ....]
    务必要让每一个问题和答案对应起来（下标位置一致）
    """
    # 问题列表
    qlist = []
    # 答案列表
    alist = []
    # 文件名称
    filename = 'data/train-v2.0.json'
    # 加载json文件
    datas = json.load(open(filename, 'r'))
    data = datas['data']
    for d in data:
        paragraph = d['paragraphs']
        for p in paragraph:
            qas = p['qas']
            for qa in qas:
                # 处理is_impossible为True时answers空
                if (not qa['is_impossible']):
                    qlist.append(qa['question'])
                    alist.append(qa['answers'][0]['text'])
    # 如果它的条件返回错误，则终止程序执行
    # 这行代码是确保长度一样
    assert len(qlist) == len(alist)
    return qlist, alist


# 读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist
qlist, alist = read_corpus()

# ### 1.2 理解数据（可视化分析/统计信息）
# 统计一下在qlist中总共出现了多少个单词？ 总共出现了多少个不同的单词(unique word)？
# 这里需要做简单的分词，对于英文我们根据空格来分词即可，其他过滤暂不考虑（只需分词）
words_qlist = dict()
for q in qlist:
    # 以空格为分词，都转为小写
    words = q.strip().split(' ')
    for w in words:
        if w.lower() in words_qlist:
            words_qlist[w.lower()] += 1
        else:
            words_qlist[w.lower()] = 1
word_total = len(words_qlist)
print(word_total)

# 统计一下qlist中出现1次，2次，3次... 出现的单词个数， 然后画一个plot. 这里的x轴是单词出现的次数（1，2，3，..)， y轴是单词个数。
# 从左到右分别是 出现1次的单词数，出现2次的单词数，出现3次的单词数
import matplotlib.pyplot as plt
import numpy as np

# counts：key出现N次，value：出现N次词有多少
counts = dict()
for w, c in words_qlist.items():
    if c in counts:
        counts[c] += 1
    else:
        counts[c] = 1
# 以histogram画图
fig, ax = plt.subplots()
ax.hist(counts.values(), bins=np.arange(0, 250, 25), histtype='step', alpha=0.6, label="counts")
ax.legend()
ax.set_xlim(0, 250)
ax.set_yticks(np.arange(0, 500, 50))
plt.show()

# #### 1.3 文本预处理
# 此部分需要做文本方面的处理。 以下是可以用到的一些方法：
#
# - 1. 停用词过滤 （去网上搜一下 "english stop words list"，会出现很多包含停用词库的网页，或者直接使用NLTK自带的）
# - 2. 转换成lower_case： 这是一个基本的操作
# - 3. 去掉一些无用的符号： 比如连续的感叹号！！！， 或者一些奇怪的单词。
# - 4. 去掉出现频率很低的词：比如出现次数少于10,20.... （想一下如何选择阈值）
# - 5. 对于数字的处理： 分词完只有有些单词可能就是数字比如44，415，把所有这些数字都看成是一个单词，这个新的单词我们可以定义为 "#number"
# - 6. lemmazation： 在这里不要使用stemming， 因为stemming的结果有可能不是valid word。

import nltk
from nltk.corpus import stopwords
import codecs
import re


# 去掉一些无用的符号
def tokenizer(ori_list):
    # 利用正则表达式去掉无用的符号
    # compile 函数用于编译正则表达式，[]用来表示一组字符
    # \s匹配任意空白字符，等价于 [\t\n\r\f]。
    SYMBOLS = re.compile('[\s;\"\",.!?\\/\[\]\{\}\(\)-]+')
    new_list = []
    for q in ori_list:
        # split 方法按照能够匹配的子串将字符串分割后返回列表
        words = SYMBOLS.split(q.lower().strip())
        new_list.append(' '.join(words))
    return new_list


# 去掉question的停用词
def removeStopWord(ori_list):
    new_list = []
    # nltk中stopwords包含what等，但是在QA问题中，这算关键词，所以不看作关键词
    restored = ['what', 'when', 'which', 'how', 'who', 'where']
    # nltk中自带的停用词库
    english_stop_words = list(
        set(stopwords.words('english')))  # ['what','when','which','how','who','where','a','an','the']
    # 将在QA问答系统中不算停用词的词去掉
    for w in restored:
        english_stop_words.remove(w)
    for q in ori_list:
        # 将每个问句的停用词去掉
        sentence = ' '.join([w for w in q.strip().split(' ') if w not in english_stop_words])
        # 将去掉停用词的问句添加至列表中
        new_list.append(sentence)
    return new_list


def removeLowFrequence(ori_list, vocabulary, thres=10):
    """
    去掉低频率的词
    :param ori_list: 预处理后的问题列表
    :param vocabulary: 词频率字典
    :param thres: 频率阈值，可以基于数据实际情况进行调整
    :return: 新的问题列表
    """
    # 根据thres筛选词表，小于thres的词去掉
    new_list = []
    for q in ori_list:
        sentence = ' '.join([w for w in q.strip().split(' ') if vocabulary[w] >= thres])
        new_list.append(sentence)
    return new_list


def replaceDigits(ori_list, replace='#number'):
    """
    将数字统一替换为replace,默认#number
    :param ori_list: 预处理后的问题列表
    :param replace:
    :return:
    """
    # 编译正则表达式：匹配1个或多个数字
    DIGITS = re.compile('\d+')
    new_list = []
    for q in ori_list:
        # re.sub用于替换字符串中的匹配项，相当于在q中查找连续的数字替换为#number
        q = DIGITS.sub(replace, q)
        # 将处理后的问题字符串添加到新列表中
        new_list.append(q)
    return new_list


def createVocab(ori_list):
    """
    创建词表，统计所有单词总数与每个单词总数
    :param ori_list:预处理后的列表
    :return:所有单词总数与每个单词总数
    """
    count = 0
    vocab_count = dict()
    for q in ori_list:
        words = q.strip().split(' ')
        count += len(words)
        for w in words:
            if w in vocab_count:
                vocab_count[w] += 1
            else:
                vocab_count[w] = 1
    return vocab_count, count


def writeFile(oriList, filename):
    """
    将处理后的问题列表写入到文件中
    :param oriList: 预处理后的问题列表
    :param filename: 文件名
    """
    with codecs.open(filename, 'w', 'utf8') as Fout:
        for q in oriList:
            Fout.write(q + u'\n')


def writeVocab(vocabulary, filename):
    """
    将词表写入到文件中
    :param vocabulary: 词表
    :param filename: 文件名
    """
    sortedList = sorted(vocabulary.items(), key=lambda d: d[1])
    with codecs.open(filename, 'w', 'utf8') as Fout:
        for (w, c) in sortedList:
            Fout.write(w + u':' + str(c) + u'\n')


# 去掉一些无用的符号
new_list = tokenizer(qlist)
# 停用词过滤
new_list = removeStopWord(new_list)
# 数字处理-将数字替换为#number
new_list = replaceDigits(new_list)
# 创建词表并统计所有单词数目
vocabulary, count = createVocab(new_list)
# 去掉低频率的词
new_list = removeLowFrequence(new_list, vocabulary, 5)
# 重新统计词频
vocab_count, count = createVocab(new_list)
# 将词表写入到文件“train.vocab”中
writeVocab(vocab_count, "train.vocab")
qlist = new_list

# ### 第二部分： 文本的表示
# 文本处理完之后，我们需要做文本表示，这里有3种方式：
#
# - 1. 使用```tf-idf vector```
# - 2. 使用embedding技术如```word2vec```, ```bert embedding```等

# #### 2.1 使用tf-idf表示向量
# 把```qlist```中的每一个问题的字符串转换成```tf-idf```向量, 转换之后的结果存储在```X```矩阵里。
# # ``X``的大小是： ``N* D``的矩阵。 这里``N``是问题的个数（样本个数），``D``是词典库的大小

import numpy as np


def computeTF(vocab, c):
    """
    计算每个词的词频
    :param vocab: 词频字典:键是单词，值是所有问句中单词出现的次数
    :param c: 单词总数
    :return: TF
    """
    # 初始化TF
    TF = np.ones(len(vocab))
    # 词频字典
    word2id = dict()
    # 单词字典
    id2word = dict()
    for word, fre in vocab.items():
        # 计算TF值：每个单词出现的个数/总的单词个数
        TF[len(word2id)] = 1.0 * fre / c
        id2word[len(word2id)] = word
        word2id[word] = len(word2id)
    return TF, word2id, id2word


def computeIDF(word2id, qlist):
    """
    计算IDF：log[问句总数/(包含单词t的问句总数+1)]
    :param word2id:单词字典
    :param qlist:问句列表
    :return:
    """
    IDF = np.ones(len(word2id))
    for q in qlist:
        # 去重
        words = set(q.strip().split())
        for w in words:
            # 统计单词出现在问句中的总数
            IDF[word2id[w]] += 1
    # 计算IDF
    IDF /= len(qlist)
    IDF = -1.0 * np.log2(IDF)
    return IDF


def computeSentenceEach(sentence, tfidf, word2id):
    """
    给定句子，计算句子TF-IDF,tfidf是一个1*M的矩阵,M为词表大小
    :param sentence:句子
    :param tfidf:TF-IDF向量
    :param word2id:词表
    :return:
    """
    sentence_tfidf = np.zeros(len(word2id))
    # 将问句以空格进行分割
    for w in sentence.strip().split(' '):
        if w not in word2id:
            continue
        # 碰到在词表word2id中的单词
        sentence_tfidf[word2id[w]] = tfidf[word2id[w]]
    return sentence_tfidf


def computeSentence(qlist, word2id, tfidf):
    """
    把```qlist```中的每一个问题的字符串转换成```tf-idf```向量, 转换之后的结果存储在```X```矩阵里
    :param qlist: 问题列表
    :param word2id: 词表(字典形式)
    :param tfidf: TF-IDF(与词表中的键一一对应)
    :return: X矩阵
    """
    # 对所有句子分别求tfidf
    X_tfidf = np.zeros((len(qlist), len(word2id)))
    for i, q in enumerate(qlist):
        X_tfidf[i] = computeSentenceEach(q, tfidf, word2id)
        # print(X_tfidf[i])
    return X_tfidf


# 计算每个词的TF,词表字典
TF, word2id, id2word = computeTF(vocab_count, count)
# 计算IDF：log[问句总数/(包含单词t的问句总数+1)]
IDF = computeIDF(word2id, qlist)
# 用TF，IDF计算最终的tf-idf,定义一个tf-idf的vectorizer
vectorizer = np.multiply(TF, IDF)
# 把```qlist```中的每一个问题的字符串转换成```tf-idf```向量, 转换之后的结果存储在```X```矩阵里
X_tfidf = computeSentence(qlist, word2id, vectorizer)

# #### 2.2 使用wordvec + average pooling
# 词向量方面需要下载： https://nlp.stanford.edu/projects/glove/ （请下载``glove.6B.zip``），并使用``d=200``的词向量（200维）。
# 国外网址如果很慢，可以在百度上搜索国内服务器上的。 每个词向量获取完之后，即可以得到一个句子的向量。 我们通过``average pooling``来实现句子的向量。
# 基于Glove向量获取句子向量
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def loadEmbedding(filename):
    """
    加载glove模型，转化为word2vec，再加载到word2vec模型
    这两种模型形式上是一样的，在数据的保存形式上有略微的差异
    :param filename: glove文件
    :return: word2vec模型
    """
    word2vec_temp_file = 'word2vec_temp.txt'
    # 加载glove模型，转化为word2vec
    glove2word2vec(filename, word2vec_temp_file)
    # 再加载到word2vec模型
    model = KeyedVectors.load_word2vec_format(word2vec_temp_file)
    return model


def computeGloveSentenceEach(sentence, embedding):
    """
    :param sentence:问题
    :param embedding:wordvec模型
    :return:
    """
    # 查找句子中每个词的embedding,将所有embedding进行加和求均值
    emb = np.zeros(200)
    words = sentence.strip().split(' ')
    for w in words:
        # 如果单词是新词，则重命名为'unknown'
        if w not in embedding:
            # 没有lookup的即为unknown
            w = 'unknown'
        # 将所有embedding进行加和求均值
        emb += embedding.get_vector(w)
        # emb += embedding[w]
    return emb / len(words)


def computeGloveSentence(qlist, embedding):
    """
    对每一个句子来构建句子向量
    :param qlist:问题列表
    :param embedding:word2vec模型
    :return:
    """
    # 对每一个句子进行求均值的embedding
    X_w2v = np.zeros((len(qlist), 200))
    for i, q in enumerate(qlist):
        # 编码每一个问题
        X_w2v[i] = computeGloveSentenceEach(q, embedding)
        # print(X_w2v)
    return X_w2v


# 加载到word2vec模型
# 这是 D*H的矩阵，这里的D是词典库的大小， H是词向量的大小。 这里面我们给定的每个单词的词向量，
emb = loadEmbedding('data/glove.6B.200d.txt')
# 初始化完emb之后就可以对每一个句子来构建句子向量了，这个过程使用average pooling来实现
X_w2v = computeGloveSentence(qlist, emb)

# #### 2.3 使用BERT + average pooling
# 我们不做任何的训练，而是直接使用已经训练好的BERT embedding. 为了获取BERT-embedding，
# 可以直接下载已经训练好的模型从而获得每一个单词的向量。可以从这里获取： https://github.com/imgarylai/bert-embedding ,
# 请使用```bert_12_768_12```	当然，你也可以从其他source获取也没问题，只要是合理的词向量。

# TODO 基于BERT的句子向量计算
from bert_embedding import BertEmbedding

sentence_embedding = np.ones((len(qlist), 768))
# 加载Bert模型，model，dataset_name,须指定
bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_cased')
# 查询所有句子的Bert  embedding
all_embedding = bert_embedding(qlist, 'sum')
for q in qlist:
    for i in range(len(all_embedding)):
        print(all_embedding[i][1])
        # 将每一个句子中每一个词的向量进行拼接求平均得到句子向量
        sentence_embedding[i] = np.sum(all_embedding[i][1], axis=0) / len(q.strip().split(' '))
        if i == 0:
            print(sentence_embedding[i])

# 每一个句子的向量结果存放在X_bert矩阵里。行数为句子的总个数，列数为一个句子embedding大小。
X_bert = sentence_embedding

# ### 第三部分： 相似度匹配以及搜索
# 在这部分里，我们需要把用户每一个输入跟知识库里的每一个问题做一个相似度计算，从而得出最相似的问题。但对于这个问题，时间复杂度其实很高，所以我们需要结合倒排表来获取相似度最高的问题，从而获得答案。

# In[ ]:


# #### 3.1 tf-idf + 余弦相似度
# 我们可以直接基于计算出来的``tf-idf``向量，计算用户最新问题与库中存储的问题之间的相似度，从而选择相似度最高的问题的答案。
# 这个方法的复杂度为``O(N)``， ``N``是库中问题的个数。

import queue as Q

# 优先级队列实现大顶堆Heap,每次输出都是相似度最大值
que = Q.PriorityQueue()


def cosineSimilarity(vec1, vec2):
    # 定义余弦相似度,余弦相似度越大，两个向量越相似
    return np.dot(vec1, vec2.T) / (np.sqrt(np.sum(vec1 ** 2)) * np.sqrt(np.sum(vec2 ** 2)))


def get_top_results_tfidf_noindex(query):
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。
    :param query:用户新问题
    :return:
    """
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 对于用户的输入 query 首先做一系列的预处理(上面提到的方法)，然后再转换成tf-idf向量（利用上面的vectorizer)
    2. 计算跟每个库里的问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    top = 5
    # 将用户输入的新问题用tf-idf来表示
    query_tfidf = computeSentenceEach(query.lower(), vectorizer, word2id)
    for i, vec in enumerate(X_tfidf):
        # 计算原问题与用户输入的新问题的相似度
        result = cosineSimilarity(vec, query_tfidf)
        # print(result)
        # 存放到大顶堆里面
        que.put((-1 * result, i))
    i = 0
    top_idxs = []
    while (i < top and not que.empty()):
        # top_idxs存放相似度最高的（存在qlist里的）问题的下标
        top_idxs.append(que.get()[1])
        i += 1
    print(top_idxs)
    # 返回相似度最高的问题对应的答案，作为TOP5答案
    return np.array(alist)[top_idxs]


# 给定用户输入的问题 query, 返回最有可能的TOP 5问题的答案。
results = get_top_results_tfidf_noindex('In what city and state did Beyonce  grow up')
print(results)

# #### 3.2 倒排表的创建
# 倒排表的创建其实很简单，最简单的方法就是循环所有的单词一遍，然后记录每一个单词所出现的文档，然后把这些文档的ID保存成list即可。我们可以定义一个类似于```hash_map```, 比如 ``inverted_index = {}``， 然后存放包含每一个关键词的文档出现在了什么位置，也就是，通过关键词的搜索首先来判断包含这些关键词的文档（比如出现至少一个），然后对于candidates问题做相似度比较。

word_doc = dict()
# key:word,value:包含该词的句子序号的列表
for i, q in enumerate(qlist):
    words = q.strip().split(' ')
    for w in set(words):
        if w not in word_doc:
            # 没在word_doc中的，建立一个空listi
            word_doc[w] = set([])
        word_doc[w] = word_doc[w] | set([i])

# 定一个一个简单的倒排表，是一个map结构。 循环所有qlist一遍就可以
inverted_idx = word_doc

# #### 3.3 语义相似度
# 读取语义相关的单词
import codecs


def get_related_words(filename):
    """
    从预处理的相似词的文件加载相似词信息
    文件格式w1 w2 w3..w11,其中w1为原词，w2-w11为w1的相似词
    :param filename: 文件名
    :return:
    """
    related_words = {}
    with codecs.open(filename, 'r', 'utf8') as Fin:
        lines = Fin.readlines()
    for line in lines:
        words = line.strip().split(' ')
        # 键为原词，值为相似词
        related_words[words[0]] = words[1:]
    return related_words


# 从预处理的相似词的文件加载相似词信息
related_words = get_related_words('data/related_words.txt')

# #### 3.4 利用倒排表搜索
# 在这里，我们使用倒排表先获得一批候选问题，然后再通过余弦相似度做精准匹配，这样一来可以节省大量的时间。搜索过程分成两步：
# - 使用倒排表把候选问题全部提取出来。首先，对输入的新问题做分词等必要的预处理工作，然后对于句子里的每一个单词，从``related_words``里提取出跟它意思相近的top 10单词， 然后根据这些top词从倒排表里提取相关的文档，把所有的文档返回。 这部分可以放在下面的函数当中，也可以放在外部。
# - 然后针对于这些文档做余弦相似度的计算，最后排序并选出最好的答案。

import queue as Q


def cosineSimilarity(vec1, vec2):
    # 定义余弦相似度
    return np.dot(vec1, vec2.T) / (np.sqrt(np.sum(vec1 ** 2)) * np.sqrt(np.sum(vec2 ** 2)))


def getCandidate(query):
    """
    根据查询句子中每个词及其10个相似词所在的序号列表，求交集
    :param query: 问题
    :return:
    """
    searched = set()
    for w in query.strip().split(' '):
        # 如果单词不在word2id中或者不在倒排表中
        if w not in word2id or w not in inverted_idx:
            continue
        # 搜索原词所在的序号列表
        if len(searched) == 0:
            searched = set(inverted_idx[w])
        else:
            searched = searched & set(inverted_idx[w])
        # 搜索相似词所在的列表
        if w in related_words:
            for similar in related_words[w]:
                searched = searched & set(inverted_idx[similar])
    return searched


def get_top_results_tfidf(query):
    """
    基于TF-IDF,给定用户输入的问题 query, 返回最有可能的TOP 5问题。
    这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate （需要使用related_words).
    2. 对于候选文档，计算跟输入问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    top = 5
    # 计算给定句子的TF-IDF
    query_tfidf = computeSentenceEach(query, vectorizer, word2id)
    # 优先级队列实现大顶堆Heap,每次输出都是相似度最大值
    results = Q.PriorityQueue()
    # 利用倒排索引表获取相似问题
    searched = getCandidate(query)
    # print(len(searched))
    for candidate in searched:
        # 计算candidate与query的余弦相似度
        result = cosineSimilarity(query_tfidf, X_tfidf[candidate])
        # 优先级队列中保存相似度和对应的candidate序号
        # -1保证降序
        results.put((-1 * result, candidate))
    i = 0
    # top_idxs存放相似度最高的（存在qlist里的）问题的索引
    top_idxs = []
    # hint: 利用priority queue来找出top results.
    while i < top and not results.empty():
        top_idxs.append(results.get()[1])
        i += 1
    # 返回相似度最高的问题对应的答案，作为TOP5答案
    return np.array(alist)[top_idxs]


def get_top_results_w2v(query):
    """
    基于word2vec，给定用户输入的问题 query, 返回最有可能的TOP 5问题。
    这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate （需要使用related_words).
    2. 对于候选文档，计算跟输入问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    # embedding用glove
    top = 5
    # 利用glove将问题进行编码
    query_emb = computeGloveSentenceEach(query, emb)
    # 优先级队列实现大顶堆Heap,每次输出都是相似度最大值
    results = Q.PriorityQueue()
    # 利用倒排索引表获取相似问题
    searched = getCandidate(query)
    for candidate in searched:
        # 计算candidate与query的余弦相似度
        result = cosineSimilarity(query_emb, X_w2v[candidate])
        # 优先级队列中保存相似度和对应的candidate序号
        # -1保证降序
        results.put((-1 * result, candidate))
    # top_idxs存放相似度最高的（存在qlist里的）问题的索引
    top_idxs = []
    i = 0
    # hint: 利用priority queue来找出top results.
    while i < top and not results.empty():
        top_idxs.append(results.get()[1])
        i += 1
    # 返回相似度最高的问题对应的答案，作为TOP5答案
    return np.array(alist)[top_idxs]


def get_top_results_bert(query):
    """
    给定用户输入的问题 query, 返回最有可能的TOP 5问题。
    这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate （需要使用related_words).
    2. 对于候选文档，计算跟输入问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    top = 5
    # embedding用Bert embedding
    query_emb = np.sum(bert_embedding([query], 'sum')[0][1], axis=0) / len(query.strip().split())
    # 优先级队列实现大顶堆Heap,每次输出都是相似度最大值
    results = Q.PriorityQueue()
    # 利用倒排索引表获取相似问题
    searched = getCandidate(query)
    for candidate in searched:
        # 计算candidate与query的余弦相似度
        result = cosineSimilarity(query_emb, X_bert[candidate])
        # 优先级队列中保存相似度和对应的candidate序号
        # -1保证降序
        results.put((-1 * result, candidate))
    # top_idxs存放相似度最高的（存在qlist里的）问题的索引
    top_idxs = []
    i = 0
    # hint: 利用priority queue来找出top results.
    while i < top and not results.empty():
        top_idxs.append(results.get()[1])
        i += 1
    # 返回相似度最高的问题对应的答案，作为TOP5答案
    return np.array(alist)[top_idxs]


# ### 4. 拼写纠错
# 4.1 训练一个语言模型
# 在这里，我们使用``nltk``自带的``reuters``数据来训练一个语言模型。 使用``add-one smoothing``

from nltk.corpus import reuters
import numpy as np
import codecs

# 读取语料库的数据
categories = reuters.categories()
corpus = reuters.sents(categories=categories)
# 循环所有的语料库并构建bigram probability.
# bigram[word1][word2]: 在word1出现的情况下下一个是word2的概率。
new_corpus = []
for sent in corpus:
    # 句子前后加入<s>,</s>表示开始和结束
    new_corpus.append(['<s> '] + sent + [' </s>'])
print(new_corpus[0])
word2id = dict()
id2word = dict()
# 构建word2id与id2word
for sent in new_corpus:
    for w in sent:
        w = w.lower()
        if w in word2id:
            continue
        id2word[len(word2id)] = w
        word2id[w] = len(word2id)
# 词表尺寸
vocab_size = len(word2id)
# 初始化
# 统计单个词出现的概率
count_uni = np.zeros(vocab_size)
# 统计两个词相邻出现的概率
count_bi = np.zeros((vocab_size, vocab_size))
for sent in new_corpus:
    for i, w in enumerate(sent):
        w = w.lower()
        count_uni[word2id[w]] += 1
        if i < len(sent) - 1:
            count_bi[word2id[w], word2id[sent[i + 1].lower()]] += 1
print("unigram done")
# 建立bigram模型
bigram = np.zeros((vocab_size, vocab_size))
# 计算bigram LM
# 有bigram统计值的加一除以|vocab|+uni统计值，没有统计值,1 除以 |vocab|+uni统计值
for i in range(vocab_size):
    for j in range(vocab_size):
        if count_bi[i, j] == 0:
            bigram[i, j] = 1.0 / (vocab_size + count_uni[i])
        else:
            bigram[i, j] = (1.0 + count_bi[i, j]) / (vocab_size + count_uni[i])


# 得到bigram概率
def checkLM(word1, word2):
    if word1.lower() in word2id and word2.lower() in word2id:
        return bigram[word2id[word1.lower()], word2id[word2.lower()]]
    else:
        return 0.0


# #### 4.2 构建Channel Probs
# 基于``spell_errors.txt``文件构建``channel probability``, 其中$channel[c][s]$表示正确的单词$c$被写错成$s$的概率。
channel = {}
# 读取文件，格式为w1:w2,w3..
# w1为正确词，w2,w3...为错误词
# 没有给出不同w2-wn的概率，暂时按等概率处理
for line in open('spell-errors.txt'):
    # 按':'为分隔符进行分割
    (correct, error) = line.strip().split(':')
    # 错误单词按照‘,’进行分割
    errors = error.split(',')
    errorProb = dict()
    for e in errors:
        # 键：错误单词；值：错误概率
        errorProb[e.strip()] = 1.0 / len(errors)
    # 键：正确单词，值：错误字典
    channel[correct.strip()] = errorProb


# print(channel)


# #### 4.3 根据错别字生成所有候选集合
# 给定一个错误的单词，首先生成跟这个单词距离为1或者2的所有的候选集合。

def filter(words):
    # 将不在词表中的词过滤
    new_words = []
    for w in words:
        if w in word2id:
            new_words.append(w)
    return set(new_words)


def generate_candidates1(word):
    # 生成DTW距离为1的词，
    # 对于英语来说，插入，替换，删除26个字母
    chars = 'abcdefghijklmnopqrstuvwxyz'
    words = set([])
    # insert 1
    words = set(word[0:i] + chars[j] + word[i:] for i in range(len(word)) for j in range(len(chars)))
    # sub 1
    words = words | set(word[0:i] + chars[j] + word[i + 1:] for i in range(len(word)) for j in range(len(chars)))
    # delete 1
    words = words | set(word[0:i] + word[i + 1:] for i in range(len(chars)))
    # 交换相邻
    # print(set(word[0:i - 1] + word[i] + word[i - 1] + word[i + 1:] for i in range(1,len(word))))
    words = words | set(word[0:i - 1] + word[i] + word[i - 1] + word[i + 1:] for i in range(1, len(word)))
    # 将不在词表中的词去掉
    words = filter(words)
    # 去掉word本身
    if word in words:
        words.remove(word)
    return words


def generate_candidates(word):
    # 基于拼写错误的单词，生成跟它的编辑距离为1或者2的单词，并通过词典库的过滤，只留写法上正确的单词。
    words = generate_candidates1(word)
    words2 = set([])
    for word in words:
        # 将距离为1词，再分别计算距离为1的词，
        # 作为距离为2的词候选
        words2 = generate_candidates1(word)
    # 过滤掉不在词表中的词
    words2 = filter(words)
    # 距离为1，2的词合并列表
    words = words | words2
    return words


# 生成候选集合
words = generate_candidates('strat')
print(words)

# #### 4.4 给定一个输入，如果有错误需要纠正
# 给定一个输入``query``, 如果这里有些单词是拼错的，就需要把它纠正过来。这部分的实现可以简单一点： 对于``query``分词，然后把分词后的每一个单词在词库里面搜一下，假设搜不到的话可以认为是拼写错误的! 人如果拼写错误了再通过``channel``和``bigram``来计算最适合的候选。
import numpy as np
import queue as Q


def word_corrector(word, context):
    """
    在候选词中基于上下文找到概率最大的词
    :param word:
    :param context:
    :return:
    """
    word = word.lower()
    # 生成跟这个单词距离为1或者2的所有的候选集合
    candidate = generate_candidates(word)
    if len(candidate) == 0:
        return word
    # 优先级队列实现大顶堆Heap,每次输出都是相似度最大值
    correctors = Q.PriorityQueue()
    # 通过``channel``和``bigram``来计算最适合的候选
    for w in candidate:
        # 如果候选单词在channel中并且正确单词在它的错误单词字典中，候选单词，文本的第一个第二个单词均在词表中
        if w in channel and word in channel[w] and w in word2id and context[0].lower() in word2id and context[
            1].lower() in word2id:
            # 相乘转化为log是相加
            # $c^* = \text{argmax}_{c\in candidates} ~~p(c|s) = \text{argmax}_{c\in candidates} ~~p(s|c)p(c)$
            # 其中channel部分是p(c),bigram部分是p(s|c)
            probility = np.log(channel[w][word] + 0.0001) + np.log(
                bigram[word2id[context[0].lower()], word2id[w]]) + np.log(
                bigram[word2id[context[1].lower()], word2id[w]])
            # 加入大顶堆
            correctors.put((-1 * probility, w))
    if correctors.empty():
        return word
    # 返回最大概率的候选词
    return correctors.get()[1]


word = word_corrector('strat', ('to', 'in'))
print(word)


def spell_corrector(line):
    """
    对拼写错误进行修正
    1. 首先做分词，然后把``line``表示成``tokens``
    2. 循环每一token, 然后判断是否存在词库里。如果不存在就意味着是拼写错误的，需要修正。
       修正的过程就使用上述提到的``noisy channel model``, 然后从而找出最好的修正之后的结果。
    :param line:
    :return:
    """
    new_words = []
    words = ['<s>'] + line.strip().lower().split(' ') + ['</s>']
    for i, word in enumerate(words):
        # 如果索引是尾部索引为break，因为尾部是'</s>'
        if i == len(words) - 1:
            break
        word = word.lower()
        if word not in word2id:
            # 认为错误，需要修正，句子前后加了<s>,</s>
            # 不在词表中词,肯定位于[1,len - 2]之间
            # 如果单词不在词表中，则说明需要修正，将修正后的词添加到new_words中
            new_words.append(word_corrector(word, (words[i - 1].lower(), words[i + 1].lower())))
        else:
            new_words.append(word)
    # 修正后的结果
    newline = ' '.join(new_words[1:])
    # 返回修正之后的结果，假如用户输入没有问题，那这时候``newline = line``
    return newline


sentence = spell_corrector('When did Beyonce strat becoming popular')
print(sentence)

# #### 4.5 基于拼写纠错算法，实现用户输入自动矫正
# 首先有了用户的输入``query``， 然后做必要的处理把句子转换成tokens的形状，然后对于每一个token比较是否是valid, 如果不是的话就进行下面的修正过程。

test_query1 = "When did Beyonce strat becoming popular"  # 拼写错误的
# result:in the late 1990s
test_query2 = "What counted for more of the poplation change"  # 拼写错误的
# result:births and deaths
test_query1 = spell_corrector(test_query1)
test_query2 = spell_corrector(test_query2)
print(test_query1)
print(test_query2)
