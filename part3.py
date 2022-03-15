#!/usr/bin/env python
# coding: utf-8

# # Import libs

# In[ ]:


import numpy as np
import pyspark
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkConf, SparkContext


# # Prepare Spark

# In[ ]:

conf = SparkConf().setAppName("part3").set("spark.driver.maxResultSize","4g")
sc = SparkContext(conf=conf)

# # Global

# In[ ]:


NUM_PARTITIONS = 160


# Define aggregate functions:

# In[ ]:


def agg_avg(t):
    y1 = t[0]
    y2 = t[1]
    key = (y1[0], y2[0])
    v1 = y1[1]
    v2 = y2[1]
    value = np.mean([v1, v2], axis=0)
    return (key, value)


def agg_min(t):
    y1 = t[0]
    y2 = t[1]
    key = (y1[0], y2[0])
    v1 = y1[1]
    v2 = y2[1]
    value = np.min([v1, v2], axis=0)
    return (key, value)


def agg_max(t):
    y1 = t[0]
    y2 = t[1]
    key = (y1[0], y2[0])
    v1 = y1[1]
    v2 = y2[1]
    value = np.max([v1, v2], axis=0)
    return (key, value)


# In[ ]:


AGG_F = agg_avg
#AGG_F = agg_min
#AGG_F = agg_max


# Define cosine similarity function:

# In[ ]:


def cos_sim(t):
    x = t[0]
    y = t[1]
    key = (x[0], y[0])
    x_v = x[1]
    y_v = y[1]
    cos_sim = dot(x_v, y_v) / (norm(x_v) * norm(y_v))
    return (key, cos_sim)


# Define threshold:

# In[ ]:


T = 0.994


# # Load Data

# In[ ]:


stock_rdd = (
    sc.textFile("/stock.csv")
    .repartition(NUM_PARTITIONS)
    .map(lambda line: line.split(","))
    .map(
        lambda row: (row[0], [(int(row[1]), float(row[2]))])
    )  # Map lines to paired RDDs (stock_name, [(rn, price)])
    .reduceByKey(lambda x, y: x + y)  # Reduce and combine time series per stock
    .map(
        lambda item: (
            item[0],
            np.array([i[1] for i in sorted(item[1], key=lambda t: t[0])]),
        )
    )  # Sort time series array by 'rn', then remove 'rn'
    .zipWithIndex()  # Include index for later joins
    .map(lambda x: (x[1], x[0]))  # Invert value and index to use index as key
)

stock_broadcasted = sc.broadcast(
    stock_rdd.collectAsMap()
)  # Broadcast and cache stock data in workers' memory


# In[ ]:


wiki_rdd = (
    sc.textFile("/wiki.csv")
    .repartition(NUM_PARTITIONS)
    .map(lambda line: line.split(","))
    .map(
        lambda row: (row[0], [(int(row[1]), float(row[2]))])
    )  # Map lines to paired RDDs (article_name, [(rn, views)])
    .reduceByKey(lambda x, y: x + y)  # Reduce and combine time series per article
    .map(
        lambda item: (
            item[0],
            np.array([i[1] for i in sorted(item[1], key=lambda t: t[0])]),
        )
    )  # Sort time series array by 'rn', then remove 'rn'
    .zipWithIndex()  # Include index for later joins
    .map(lambda x: (x[1], x[0]))  # Invert value and index to use index as key
)

wiki_broadcasted = sc.broadcast(
    wiki_rdd.collectAsMap()
)  # Broadcast and cache wiki data in workers' memory


# # Calculate AGG_F(Y1, Y2)
# 
# https://stackoverflow.com/questions/38828139/spark-cartesian-product
# 
# https://umbertogriffo.gitbook.io/apache-spark-best-practices-and-tuning/rdd/joining-a-large-and-a-small-rdd

# In[ ]:


idxs = range(1000)  # The number here is the size of wiki dataset
indices = sc.parallelize(
    [(i, j) for i in idxs for j in idxs if i < j], NUM_PARTITIONS
)  # Generate unique indices pairs. This is equivalent to combination 1000C2


# In[ ]:


wiki_pairs = (
    indices.map(
        lambda t: (wiki_broadcasted.value[t[0]], wiki_broadcasted.value[t[1]])
    )  # Map keys from indices with values from broadcasted wiki dataset. This returns all combinations of pairs of wiki articles
    .map(AGG_F)  # apply chosen aggregate function on each combination
    .zipWithIndex()
    .map(lambda x: (x[1], x[0]))
)

wiki_pairs_broadcasted = sc.broadcast(
    wiki_pairs.collectAsMap()
)  # Broadcast and cache wiki pairs data in workers' memory


# # Calculate cos_sim(X, AGG_F(Y1, Y2))

# In[ ]:


idxs1 = sc.parallelize(
    range(1000), NUM_PARTITIONS
)  # The number here is the size of stock dataset

idxs2 = sc.parallelize(
    range(499500), NUM_PARTITIONS
)  # The number here is the size of wiki_pairs, which can be calculate by 1000C2

indices = idxs1.cartesian(idxs2).coalesce(
    NUM_PARTITIONS
)  # Generate all combinations of indices pairs


# In[ ]:


result = (
    indices.map(
        lambda t: (stock_broadcasted.value[t[0]], wiki_pairs_broadcasted.value[t[1]])
    ) # Map keys from indices with values from broadcasted wiki pairs dataset
    .map(cos_sim) # apply cosine similarity function on each combination
    .filter(lambda x: x[1] > T) # Filter using defined threshold
)


# In[ ]:


print(result.toDebugString())


# In[ ]:


print(result.collect())


# In[ ]:


sc.stop()

