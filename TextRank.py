#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark import SparkContext
from nltk import tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from pyspark.sql.functions import row_number,lit
from pyspark.sql.window import Window
from pyspark.sql import Row
import numpy as np
import pyspark
import sys
import re
import nltk
from pyspark.sql import SparkSession
import pandas
from pyspark.sql import SQLContext
import rouge
from rouge import Rouge 


# In[2]:


sc = SparkContext()


# In[3]:


spark = SparkSession.builder.appName("text_rank").getOrCreate


# In[4]:


sql = SQLContext(sc)
datafile = pandas.read_csv('dataset.csv')
data = sql.createDataFrame(datafile)
content = data.select('CONTENT')
w = Window().orderBy(lit('A'))
df = content.withColumn("row_num", row_number().over(w))
rdd = df.rdd.map(list)
items = rdd.count()
NoOfIterations = 20
nltk.download("wordnet")
nltk.download("stopwords")
sw = stopwords.words('english')


# In[5]:


def CreateNode(file):
    sentence_id = file[1]
    sentences = file[0].split(".")
    output = []
    for index, sent in enumerate(sentences):
        sent_id = str(sentence_id) + '_' + str(index)
        sent_len = len(sent.split(" "))
        if 10 < sent_len < 30:
            output.append((sent_id, sent))
    return output


# In[6]:


def FilterNode(sent):
    sentence_id, sentence = sent[0], sent[1]
    lmtz = WordNetLemmatizer()
    result = []
    output = []
    words = re.findall(r'[a-zA-Z]+', str(sentence))
    for w in words:
        if w.lower() not in sw:
            w = lmtz.lemmatize(w.lower())
            result.append(w)
    for w in result:
        if len(w) > 3:
            word = w
            output.append(word)
    return sentence_id, output


# In[7]:


def AdjecencyList(node, totalnodes):
    n,v = node[0],node[1]
    edges = {}
    for u in totalnodes:
        edge = Relation(node, u)
        if edge is not None:
            edges[edge[0]] = edge[1]
    return (n, edges)


# In[8]:


def Relation(node1, node2):
    n1,v1 = node1[0],node1[1]
    n2,v2 = node2[0],node2[1]
    if n1 != n2: 
        n_len = len(set(v1).intersection(v2))
        log_len = np.log2(len(v1)) + np.log2(len(v2))
        if log_len == 0:
            coefficient = n_len/(log_len+1)
        else:
            coefficient = n_len/(log_len)
        if coefficient != 0:
            return (n2, coefficient)


# In[9]:


def Contribution(neighbours, rank):
    output = []
    totalweight = sum(neighbours.values())
    for key,weight in neighbours.items():
        weightage = (rank*weight)/totalweight
        output.append((key, weightage))
    return output


# In[10]:


def Summary(text):
    file = CreateNode(text)
    textfile = sc.parallelize(file)
    nodes = textfile.map(lambda l: FilterNode(l))
    textfile = textfile.cache()
    total_nodes = nodes.collect()
    textrank_graph = nodes.map(lambda ver: AdjecencyList(ver, total_nodes))
    textrank_graph = textrank_graph.filter(lambda l: len(l[1]) > 0).cache()
    rank_rdd = textrank_graph.map(lambda x: (x[0],0.15))
    collection = textrank_graph.join(rank_rdd).flatMap(lambda x: Contribution(x[1][0], x[1][1]))
    for i in range(0,NoOfIterations):
        collection = textrank_graph.join(rank_rdd).flatMap(lambda x: Contribution(x[1][0], x[1][1]))
        rank_rdd = collection.reduceByKey(lambda x,y: x+y).mapValues(lambda r: 0.15 + 0.85 * r)
    sent = ""
    finalrank = rank_rdd.collect()
    count = rank_rdd.count()
    if count < 5:
        sentence_count = count
    else:
        sentence_count = 5
    result = sorted(finalrank, key=lambda x: x[1], reverse=True)
    textfile.collect()
    for j in range(0, sentence_count):
        sent = sent + textfile.lookup(result[j][0])[0].replace('\n', "")
    output = []
    output.append(sent)
    return output


# In[11]:


result = []
for i in range(1,items+1):
    result.append(Summary(rdd.take(i)[i-1]))
df = sql.createDataFrame(result, ['GENERATEDSUMMARY'])


# In[12]:


data = data.withColumnRenamed("url","URL").withColumnRenamed("category", "CATEGORY").withColumnRenamed("content", "CONTENT").withColumnRenamed("summary", "ORIGINALSUMMARY")


# In[13]:


data =  data.withColumn("row_num", row_number().over(w))
df =  df.withColumn("row_num", row_number().over(w))


# In[14]:


summary = data.join(df, data.row_num == df.row_num, 'inner').drop(df.row_num)
summary = summary.drop("row_num")


# In[15]:


TextrankSummary = summary.toPandas()


# In[16]:


TextrankSummary.to_csv('TextrankResult.csv')


# In[17]:


hypothesis = TextrankSummary["GENERATEDSUMMARY"]
reference = TextrankSummary["ORIGINALSUMMARY"]


# In[18]:


hypothesis[2]


# In[19]:


reference[2]


# In[20]:


allhypothesis = [''.join(hypothesis[0 : len(hypothesis)-1])]
allreference = [''.join(reference[0 : len(reference)-1])]


# In[21]:


def convert(list): 
    return (list[0].split()) 


# In[22]:


hypothesis_words = convert(hypothesis)
reference_words = convert(reference)
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference_words], hypothesis_words, weights = (0.5, 0.5))
print(BLEUscore)


# In[23]:


hypothesis_words = convert(allhypothesis)
reference_words = convert(allreference)
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference_words], hypothesis_words, weights = (0.5, 0.5))
print(BLEUscore)


# In[24]:


rouge = Rouge()
scores = rouge.get_scores(hypothesis[0:5], reference[0:5])
scores


# In[25]:


sys.setrecursionlimit(len(hypothesis_words)*len(reference_words)+10)


# In[26]:


rouge = Rouge()
scores = rouge.get_scores(allhypothesis, allreference)
scores


# In[ ]:




