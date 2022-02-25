# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: ''
#     name: ''
# ---

# + tags=[]
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.getOrCreate()

# +
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType
from pyspark.sql.functions import split

schema = StructType([
    StructField("stock", StringType()),
    StructField("date", DateType()),
    StructField("price", DoubleType()),
    StructField("volume", DoubleType())
])

df = spark.read.csv("data/raw/MS1.txt", schema=schema, dateFormat="MM/dd/yyyy")

split_col = split(df["stock"], "\.", 2)

df = df.withColumn("stockNo", split_col.getItem(0).cast(IntegerType())).withColumn("stockName", split_col.getItem(1)).select("stockNo", "stockName", "date", "price", "volume")

df.show(truncate=False)
# -

df.count()

# + tags=[]
#count null for each col
df_agg = df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns])
df_agg.show()
# -

df_pre = df.dropna()
df_pre.show(truncate=False)

df_pre.count()

# + tags=[]
cols = df_pre.columns
print(cols)
# -


