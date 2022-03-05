# Imports
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType

import pyspark.sql.functions as F
import datetime
import time
import numpy as np

#42.333091259002686 seconds with 5 time series for stocks dataset
#66.53 seconds with 1496 time series
#5.93 seconds for wiki dataset

start_time = time.time()

spark = SparkSession.builder.getOrCreate()

#INTERPOLATION FUNCTIONS
def date_range(date, step=1, enddate=False):
    """Return a list of 1000 dates from start date or before enddate.
    
    The list also includes 6 days prior and 6 days after the 1000 dates that will be used.
    5 days is the maximum number of sequential blank rows in our dataset. 
    Including the rows prior and after our dates mean that interpolation is always the norm and backfilling/forward filling are only 
    used when absolutely necessary and not because we filtered out the relevant prior/ following rows.
    """
    date_range = []
    if enddate == True:
        date = date - datetime.timedelta(days=999)
    
    date =  date - datetime.timedelta(days=6)
    for i in range(1012):
        date_range.append(date)
        date = date + datetime.timedelta(days=step)
    return date_range

def fill_linear_interpolation(df,id_col,order_col,value_col,num_dec):
    """
    Apply linear interpolation to dataframe to fill gaps. Backfill and forwardfill rows if no non-empty entry before/ after.
    Based on https://stackoverflow.com/questions/53077639/pyspark-interpolation-of-missing-values-in-pyspark-dataframe-observed

    :param df: spark dataframe
    :param id_col: string or list of column names to partition by the window function
    :param order_col: column to use to order by the window function
    :param value_col: column to be filled
    :param num_dec: number of decimal places to return interpolated values with

    :returns: spark dataframe updated with interpolated values
    """
    # create row number over window and a column with row number only for non missing values
    w = Window.partitionBy(id_col).orderBy(order_col)
    new_df = df.withColumn('rn_not_null',F.when(F.col(value_col).isNotNull(),F.col('rn')))

    # find last non missing value
    w_before = Window.partitionBy(id_col).orderBy(order_col).rowsBetween(Window.unboundedPreceding,-1) #checking all previous rows
    new_df = new_df.withColumn('start_val',F.last(value_col,True).over(w_before)) # returning last non negative value
    new_df = new_df.withColumn('start_rn',F.last('rn_not_null',True).over(w_before))

    # find next non missing value
    w_after = Window.partitionBy(id_col).orderBy(order_col).rowsBetween(0,Window.unboundedFollowing)
    new_df = new_df.withColumn('end_val',F.first(value_col,True).over(w_after))
    new_df = new_df.withColumn('end_rn',F.first('rn_not_null',True).over(w_after))

    # create references to gap length and current gap position
    new_df = new_df.withColumn('diff_rn',F.col('end_rn')-F.col('start_rn'))
    new_df = new_df.withColumn('curr_rn',F.col('diff_rn')-(F.col('end_rn')-F.col('rn')))

    # calculate linear interpolation value
    lin_interp_func = (F.col('start_val')+((F.col('end_val')-F.col('start_val'))/F.col('diff_rn'))*F.col('curr_rn'))
    new_df = new_df.withColumn(value_col,
                               F.when(F.col(value_col).isNull() & F.col('start_val').isNotNull() & F.col('end_val').isNotNull(), F.round(lin_interp_func,num_dec)) 
                               .when(F.col(value_col).isNull() & F.col('start_val').isNotNull() & F.col('end_val').isNull(), F.col('start_val')) 
                               .when(F.col(value_col).isNull() & F.col('start_val').isNull() & F.col('end_val').isNotNull(), F.col('end_val')) 
                               .otherwise(F.col(value_col)))
    
    new_df = new_df.drop('rn_not_null', 'start_val', 'end_val', 'start_rn', 'end_rn', 'diff_rn', 'curr_rn')
    return new_df


## STOCK DATA
schema = StructType([
    StructField("stock", StringType()),
    StructField("date", DateType()),
    StructField("price", DoubleType()),
    StructField("volume", DoubleType())
])
# Read csv used so can use schema
stocks_df = spark.read.csv("data/raw/MS1.txt", schema=schema, dateFormat="MM/dd/yyyy")

# Duplicate rows with identical data were identified so these are removed
stocks_df = stocks_df.distinct()

stocks_filtered_df = stocks_df.where(F.col('stock').contains('Nordamerika_USA-S-P'))

#Creating date dataframe with row number
date_df = spark.createDataFrame(((np.array([date_range(datetime.date(2020, 12, 31), enddate=True), range(-5,1007)])).T).tolist(), schema=["date", "rn"])

# Creating dataframe that is a combination of every date and stock
stock_distinct = stocks_filtered_df.select('stock').distinct()
# Repartitioning to improve performance
stock_date_df = date_df.repartition(200, "date").crossJoin(stock_distinct)

# Joining to original stocks data (dropping volume variable)
stocks_proc_df = stock_date_df.join(stocks_filtered_df.select('stock', 'date', 'price'), on=['stock', 'date'], how='left')

#Using 5 decimal places as this is the maximum precision used in our relevant stocks
stocks_proc_df = fill_linear_interpolation(stocks_proc_df, 'stock', 'rn', 'price', 5)

# Filter to the 1000 days we are interested in (removing extra data used for interpolation)
stocks_proc_df = stocks_proc_df.filter((F.col("rn") > 0) & (F.col("rn") < 1001))

# Saving dataset
stocks_proc_df.write.mode("overwrite").csv('data/processed/stock')

stock_runtime = time.time()
print("Time to run stock dataset: %s seconds" % (stock_runtime - start_time))


##WIKIPEDIA DATA
schema = StructType([
    StructField("article", StringType()),
    StructField("timestamp", DateType()),
    StructField("views", IntegerType())
])

# Read csv used so can use schema
wiki_df = spark.read.csv("data/raw/wikipedia.csv", schema=schema, dateFormat="yyyyMMddHH", header=True)

# Removing duplicate rows (occurring from errors saving dataframe)
wiki_df = wiki_df.distinct()

# Filtering data df to only 1000 dates we're interested in (interpolation not required for this dataset)
date_df = date_df.filter((F.col("rn") > 0) & (F.col("rn") < 1001))
# Renaming date column in date dataframe to allow join
wiki_date_df = date_df.withColumnRenamed("date","timestamp")

# Creating dataframe that is a combination of every date and article (can be optimized?)
article_distinct = wiki_df.select('article').distinct()
wiki_date_df = wiki_date_df.crossJoin(article_distinct)

# Joining to original wikipedia data
wiki_proc_df = wiki_date_df.join(wiki_df, on=['article', 'timestamp'], how='left')

# Based on this article we fill in the nulls as 0 page views
# The four concerned articles with null all have significantly below average page views confirming the likelihood of them having days with no page views
wiki_proc_df = wiki_proc_df.withColumn("views", F.when(wiki_proc_df.views.isNull(),0)
                                 .otherwise(wiki_proc_df.views))

# Saving processed data
wiki_proc_df.write.mode("overwrite").csv('data/processed/wiki')
print("Time to run wiki dataset: %s seconds" % (time.time() - stock_runtime))