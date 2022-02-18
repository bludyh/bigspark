# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from pyspark.sql import SparkSession

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
