# this is the max version, to use min/avg, simply change the udf_min in wiki_pairs to udf_max or udf_avg, also change the threshold to ensure optimal performance
# tau value (from part 3, be sure to use a slightly smaller threshold for subsets)
# AVG 0.9946
# MIN 0.9974
# MAX 0.993555

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType, ArrayType

import pyspark.sql.functions as F
import pyspark
import numpy as np
import time
import sys

spark = (
    SparkSession
    .builder
    .appName("Part2")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

## Reading in data
stock_schema = StructType([
    StructField("stock", StringType()),
    StructField("rn", IntegerType()),
    StructField("price", DoubleType())
])
df_stock = spark.read.csv("/stock.csv",header=False, schema=stock_schema)


wiki_schema = StructType([
    StructField("article", StringType()),
    StructField("rn", IntegerType()),
    StructField("views", IntegerType())
])

df_wiki = spark.read.csv("/wiki.csv",header=False, schema=wiki_schema)
df_wiki.repartition(200)

df_stock.createOrReplaceTempView("stock")
df_wiki.createOrReplaceTempView("wiki")

def get_max(x,y):
    return np.maximum(x,y).tolist()
def get_min(x,y):
    return np.minimum(x,y).tolist()
def get_avg(x,y):
    return np.mean([x,y], axis=0).tolist()
def get_norm(x):
    return float(np.linalg.norm(x))
def get_cos_sim(x,y,norm_x,norm_y):
    return float(np.dot(x, y) / (norm_x * norm_y))

spark.udf.register("udf_max", get_max, ArrayType(IntegerType()))
spark.udf.register("udf_min", get_min, ArrayType(IntegerType()))
spark.udf.register("udf_avg", get_avg, ArrayType(DoubleType()))
spark.udf.register("udf_norm", get_norm, DoubleType())
spark.udf.register("udf_cos_sim", get_cos_sim, DoubleType())

stock_list = spark.sql("""
SELECT stock, price_list, udf_norm(price_list) AS p_norm
FROM (SELECT
    stock,
    rn,
    collect_list(price) OVER (PARTITION BY stock ORDER BY rn ASC) AS price_list
  FROM stock)
WHERE rn=1000
""")

stock_list = stock_list.repartition(200)

article_list = spark.sql("""
SELECT article, views_list
FROM (SELECT
    article,
    rn,
    collect_list(views) OVER (PARTITION BY article ORDER BY rn ASC) AS views_list
  FROM wiki)
WHERE rn=1000
""")

article_list = article_list.repartition(200)
article_list.createOrReplaceTempView("article_list")

wiki_pairs =spark.sql(sqlQuery="""
SELECT /*+ BROADCAST(article_list) */  a.article AS y1_article, b.article AS y2_article, udf_min(a.views_list, b.views_list) AS y
FROM article_list as a
CROSS JOIN article_list as b
on a.article < b.article
""")

wiki_pairs = wiki_pairs.coalesce(200)

stock_list.createOrReplaceTempView("stock_list")
wiki_pairs.createOrReplaceTempView("wiki_pairs")

cos_df = spark.sql(
"""
SELECT /*+ BROADCAST(stock_join) */ stock, y1_article, y2_article, udf_cos_sim(price_list,y,stock_join.p_norm, y_norm) AS cos_sim
FROM (SELECT *, udf_norm(y) AS y_norm
FROM wiki_pairs)
CROSS JOIN (SELECT *
FROM stock_list) AS stock_join""")

cos_df = cos_df.coalesce(200)
cos_df.createOrReplaceTempView("cos_df")

spark.sql("""
SELECT *
FROM cos_df
WHERE cos_sim > 0.997
""").show()