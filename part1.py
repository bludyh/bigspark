# Imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType
from pyspark.sql import Window

import pyspark.sql.functions as F
import datetime
import time
import numpy as np

# Start Spark
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

def stock_linear_interpolation(df):
    """
    Apply linear interpolation to dataframe to fill gaps. Backfill and forwardfill rows if no non-empty entry before/ after.
    Fixed price column is rounded to 5 digits as this is the maximum precision in price in this dataset.
    """
    
    df.createOrReplaceTempView("df")
    new_df = spark.sql("""
    WITH rn_notnull_df AS (
    SELECT *, 
    CASE
        WHEN price IS NOT NULL THEN rn
        ELSE NULL
    END AS rn_notnull
    FROM df
    ),
    new_col AS (SELECT *, 
    LAST_VALUE(price) IGNORE NULLS OVER (PARTITION BY stock ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS prev_non_null_price,
LAST_VALUE(rn_notnull) IGNORE NULLS OVER (PARTITION BY stock ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS prev_non_null_rn,
    FIRST_VALUE(price) IGNORE NULLS OVER (PARTITION BY stock ORDER BY rn ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS next_non_null_price,
    FIRST_VALUE(rn_notnull) IGNORE NULLS OVER (PARTITION BY stock ORDER BY rn ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS next_non_null_rn
    FROM rn_notnull_df),
    calc AS (SELECT *, (next_non_null_rn - prev_non_null_rn) AS diff_non_null_rn, (rn - prev_non_null_rn) AS curr_diff_non_null_rn
    FROM new_col)
    
    SELECT stock, rn,
    CASE 
        WHEN price IS NOT NULL THEN price
        WHEN prev_non_null_price IS NULL AND next_non_null_price IS NOT NULL THEN next_non_null_price
        WHEN prev_non_null_price IS NOT NULL AND next_non_null_price IS NULL THEN prev_non_null_price
        ELSE ROUND(prev_non_null_price + (((next_non_null_price - prev_non_null_price)/diff_non_null_rn) * curr_diff_non_null_rn),5)
    END AS price
    FROM calc
    """)
    
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

# Filtering to S&P stocks
stocks_filtered_df = stocks_df.where(F.col('stock').contains('Nordamerika_USA-S-P'))
# Filtering to 1000 of these stocks
stocks_filtered_df = stocks_filtered_df.filter(F.col("stock").isin(stocks_filtered_df
                                                                   .select("stock").distinct().rdd.flatMap(lambda x:x).collect()[:1000]))

#Creating date dataframe with row number
date_df = spark.createDataFrame(((np.array([date_range(datetime.date(2020, 12, 31), enddate=True), range(-5,1007)])).T).tolist(), schema=["date", "rn"])

# Creating dataframe that is a combination of every date and stock
# Repartitioning to improve performance
stock_distinct = stocks_filtered_df.select('stock').distinct()
stock_date_df = date_df.repartition(20).crossJoin(stock_distinct)

# Joining to original stocks data (dropping volume variable)
# Using left join to identify dates missing price data
stocks_proc_df = stock_date_df.join(stocks_filtered_df.select('stock', 'date', 'price'), on=['stock', 'date'], how='left')

#Using 5 decimal places as this is the maximum precision used in our relevant stocks
stocks_proc_df = stock_linear_interpolation(stocks_proc_df)

# Filter to the 1000 days we are interested in (removing extra data used for interpolation)
stocks_proc_df = stocks_proc_df.filter((F.col("rn") > 0) & (F.col("rn") < 1001))
# Dropping date column
stocks_proc_df = stocks_proc_df.drop('date')

# Saving dataset
stocks_proc_df.repartition(1).orderBy('stock','rn').write.mode("overwrite").csv('data/processed/stock')


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
# Filtering to 1000 most important articles
wiki_df = wiki_df.filter(F.col("article").isin(wiki_df.select("article").distinct().rdd.flatMap(lambda x:x).collect()[:1000]))

# Filtering date_df to only 1000 dates we're interested in (interpolation not required for this dataset)
date_df = date_df.filter((F.col("rn") > 0) & (F.col("rn") < 1001))
# Renaming date column in date dataframe to allow join
wiki_date_df = date_df.withColumnRenamed("date","timestamp")

# Creating dataframe that is a combination of every date and article
article_distinct = wiki_df.select('article').distinct()
wiki_date_df = wiki_date_df.repartition(20).crossJoin(article_distinct)

# Joining to original wikipedia data
# Using left join to identify dates missing views data
wiki_proc_df = wiki_date_df.join(wiki_df, on=['article', 'timestamp'], how='left')

# Based on API documentation and our analysis the missing views correspond to a day with 0 views
# Null rows therefore filled with 0
wiki_proc_df = wiki_proc_df.withColumn("views", F.when(wiki_proc_df.views.isNull(),0)
                                 .otherwise(wiki_proc_df.views))
# Dropping timestamp column
wiki_proc_df = wiki_proc_df.drop('timestamp')

# Saving processed data to csv for part 3
wiki_proc_df.repartition(1).orderBy('article','rn').write.mode("overwrite").csv('data/processed/wiki')
# Imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType
from pyspark.sql import Window

import pyspark.sql.functions as F
import datetime
import time
import numpy as np

# Start Spark
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

def linear_interpolation(df,id_col,order_col,value_col,num_dec):
    """
    Apply linear interpolation to dataframe to fill gaps. Backfill and forwardfill rows if no non-empty entry before/ after.
    
    NEED TO UPDATE PRICE COLUMN
    IF ALL EMPTY BEFORE BACKFILL/ AFTER FORWARDFILL/ IN BETWEEN INTERPOLATE
    
    CASE WHEN NULL THEN +1 WHEN NON NULL AGAIN THEN 0
    """
     
    # create row number over window and a column with row number only for non missing values
    w = Window.partitionBy('stock').orderBy('rn')
    new_df = df.withColumn('rn_not_null',F.when(F.col(value_col).isNotNull(),F.col('rn')))

    # find last non missing value
    w_before = Window.partitionBy(id_col).orderBy(order_col).rowsBetween(Window.unboundedPreceding,-1) #checking all previous rows
    new_df = new_df.withColumn('start_val',F.last(value_col,True).over(w_before)) # returning last non negative value
    new_df = new_df.withColumn('start_rn',F.last('rn_not_null',True).over(w_before)) # returning last non negative row number

    # find next non missing value
    w_after = Window.partitionBy(id_col).orderBy(order_col).rowsBetween(0,Window.unboundedFollowing) # checking all following rows
    new_df = new_df.withColumn('end_val',F.first(value_col,True).over(w_after)) # returning next non negative value
    new_df = new_df.withColumn('end_rn',F.first('rn_not_null',True).over(w_after)) # returning next non negative row number

    # create references to gap length and current gap position
    new_df = new_df.withColumn('diff_rn',F.col('end_rn')-F.col('start_rn')) # calculating number of empty rows between last non negative value and next non negative value
    new_df = new_df.withColumn('curr_rn',F.col('diff_rn')-(F.col('end_rn')-F.col('rn'))) # calculating number of empty rows between current row and next non negative value

    # calculate linear interpolation value
    lin_interp_func = (F.col('start_val')+((F.col('end_val')-F.col('start_val'))/F.col('diff_rn'))*F.col('curr_rn'))
    
    # fill blanks with linear interpolation value if non-empty value before and after
    # back-fill blanks with end_val if no non-empty value before
    # forward-fill blanks with start_val if no non-empty value after
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

# Filtering to S&P stocks
stocks_filtered_df = stocks_df.where(F.col('stock').contains('Nordamerika_USA-S-P'))
# Filtering to 1000 of these stocks
stocks_filtered_df = stocks_filtered_df.filter(F.col("stock").isin(stocks_filtered_df
                                                                   .select("stock").distinct().rdd.flatMap(lambda x:x).collect()[:1000]))

#Creating date dataframe with row number
date_df = spark.createDataFrame(((np.array([date_range(datetime.date(2020, 12, 31), enddate=True), range(-5,1007)])).T).tolist(), schema=["date", "rn"])

# Creating dataframe that is a combination of every date and stock
# Repartitioning to improve performance
stock_distinct = stocks_filtered_df.select('stock').distinct()
stock_date_df = date_df.repartition(20).crossJoin(stock_distinct)

# Joining to original stocks data (dropping volume variable)
# Using left join to identify dates missing price data
stocks_proc_df = stock_date_df.join(stocks_filtered_df.select('stock', 'date', 'price'), on=['stock', 'date'], how='left')

#Using 5 decimal places as this is the maximum precision used in our relevant stocks
stocks_proc_df = linear_interpolation(stocks_proc_df, 'stock', 'rn', 'price', 5)

# Filter to the 1000 days we are interested in (removing extra data used for interpolation)
stocks_proc_df = stocks_proc_df.filter((F.col("rn") > 0) & (F.col("rn") < 1001))
# Dropping date column
stocks_proc_df = stocks_proc_df.drop('date')

# Saving dataset
stocks_proc_df.repartition(1).orderBy('stock','rn').write.mode("overwrite").csv('data/processed/stock')

# Changing from storing each data point on row basis to as a list
stocks_proc_df = stocks_proc_df.withColumn("price_list", F.collect_list("price").over(Window.partitionBy("stock").orderBy("rn"))) \
    .filter(F.col("rn") == 1000).drop("rn", "price")
stocks_proc_df = stocks_proc_df.withColumn("price_list_string", F.col("price_list").cast(StringType())).drop("price_list")

# Saving processed data as parquet for part 1
stocks_proc_df.repartition(1).write.mode("overwrite").csv('data/processed/stock_list')


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
# Filtering to 1000 most important articles
wiki_df = wiki_df.filter(F.col("article").isin(wiki_df.select("article").distinct().rdd.flatMap(lambda x:x).collect()[:1000]))

# Filtering date_df to only 1000 dates we're interested in (interpolation not required for this dataset)
date_df = date_df.filter((F.col("rn") > 0) & (F.col("rn") < 1001))
# Renaming date column in date dataframe to allow join
wiki_date_df = date_df.withColumnRenamed("date","timestamp")

# Creating dataframe that is a combination of every date and article
article_distinct = wiki_df.select('article').distinct()
wiki_date_df = wiki_date_df.repartition(20).crossJoin(article_distinct)

# Joining to original wikipedia data
# Using left join to identify dates missing views data
wiki_proc_df = wiki_date_df.join(wiki_df, on=['article', 'timestamp'], how='left')

# Based on API documentation and our analysis the missing views correspond to a day with 0 views
# Null rows therefore filled with 0
wiki_proc_df = wiki_proc_df.withColumn("views", F.when(wiki_proc_df.views.isNull(),0)
                                 .otherwise(wiki_proc_df.views))
# Dropping timestamp column
wiki_proc_df = wiki_proc_df.drop('timestamp')

# Saving processed data to csv for part 3
wiki_proc_df.repartition(1).orderBy('article','rn').write.mode("overwrite").csv('data/processed/wiki')

# Changing from storing each data point on row basis to as a list
wiki_proc_df = wiki_proc_df.withColumn("views_list", F.collect_list("views").over(Window.partitionBy("article").orderBy("rn"))) \
    .filter(F.col("rn") == 1000).drop("rn", "views")
wiki_proc_df = wiki_proc_df.withColumn("views_list_string", F.col("views_list").cast(StringType())).drop("views_list")

# Saving processed data as parquet for part 1
wiki_proc_df.repartition(1).write.mode("overwrite").csv('data/processed/wiki_list')# Imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType
from pyspark.sql import Window

import pyspark.sql.functions as F
import datetime
import time
import numpy as np

# Start Spark
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

def stock_linear_interpolation(df):
    """
    Apply linear interpolation to dataframe to fill gaps. Backfill and forwardfill rows if no non-empty entry before/ after.
    Fixed price column is rounded to 5 digits as this is the maximum precision in price in this dataset.
    """
    
    df.createOrReplaceTempView("df")
    new_df = spark.sql("""
    WITH rn_notnull_df AS (
    SELECT *, 
    CASE
        WHEN price IS NOT NULL THEN rn
        ELSE NULL
    END AS rn_notnull
    FROM df
    ),
    new_col AS (SELECT *, 
    LAST_VALUE(price) IGNORE NULLS OVER (PARTITION BY stock ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS prev_non_null_price,
LAST_VALUE(rn_notnull) IGNORE NULLS OVER (PARTITION BY stock ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS prev_non_null_rn,
    FIRST_VALUE(price) IGNORE NULLS OVER (PARTITION BY stock ORDER BY rn ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS next_non_null_price,
    FIRST_VALUE(rn_notnull) IGNORE NULLS OVER (PARTITION BY stock ORDER BY rn ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS next_non_null_rn
    FROM rn_notnull_df),
    calc AS (SELECT *, (next_non_null_rn - prev_non_null_rn) AS diff_non_null_rn, (rn - prev_non_null_rn) AS curr_diff_non_null_rn
    FROM new_col)
    
    SELECT stock, rn,
    CASE 
        WHEN price IS NOT NULL THEN price
        WHEN prev_non_null_price IS NULL AND next_non_null_price IS NOT NULL THEN next_non_null_price
        WHEN prev_non_null_price IS NOT NULL AND next_non_null_price IS NULL THEN prev_non_null_price
        ELSE ROUND(prev_non_null_price + (((next_non_null_price - prev_non_null_price)/diff_non_null_rn) * curr_diff_non_null_rn),5)
    END AS price
    FROM calc
    """)
    
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

# Filtering to S&P stocks
stocks_filtered_df = stocks_df.where(F.col('stock').contains('Nordamerika_USA-S-P'))
# Filtering to 1000 of these stocks
stocks_filtered_df = stocks_filtered_df.filter(F.col("stock").isin(stocks_filtered_df
                                                                   .select("stock").distinct().rdd.flatMap(lambda x:x).collect()[:1000]))

#Creating date dataframe with row number
date_df = spark.createDataFrame(((np.array([date_range(datetime.date(2020, 12, 31), enddate=True), range(-5,1007)])).T).tolist(), schema=["date", "rn"])

# Creating dataframe that is a combination of every date and stock
# Repartitioning to improve performance
stock_distinct = stocks_filtered_df.select('stock').distinct()
stock_date_df = date_df.repartition(20).crossJoin(stock_distinct)

# Joining to original stocks data (dropping volume variable)
# Using left join to identify dates missing price data
stocks_proc_df = stock_date_df.join(stocks_filtered_df.select('stock', 'date', 'price'), on=['stock', 'date'], how='left')

#Using 5 decimal places as this is the maximum precision used in our relevant stocks
stocks_proc_df = stock_linear_interpolation(stocks_proc_df)

# Filter to the 1000 days we are interested in (removing extra data used for interpolation)
stocks_proc_df = stocks_proc_df.filter((F.col("rn") > 0) & (F.col("rn") < 1001))
# Dropping date column
stocks_proc_df = stocks_proc_df.drop('date')

# Saving dataset
stocks_proc_df.repartition(1).orderBy('stock','rn').write.mode("overwrite").csv('data/processed/stock')


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
# Filtering to 1000 most important articles
wiki_df = wiki_df.filter(F.col("article").isin(wiki_df.select("article").distinct().rdd.flatMap(lambda x:x).collect()[:1000]))

# Filtering date_df to only 1000 dates we're interested in (interpolation not required for this dataset)
date_df = date_df.filter((F.col("rn") > 0) & (F.col("rn") < 1001))
# Renaming date column in date dataframe to allow join
wiki_date_df = date_df.withColumnRenamed("date","timestamp")

# Creating dataframe that is a combination of every date and article
article_distinct = wiki_df.select('article').distinct()
wiki_date_df = wiki_date_df.repartition(20).crossJoin(article_distinct)

# Joining to original wikipedia data
# Using left join to identify dates missing views data
wiki_proc_df = wiki_date_df.join(wiki_df, on=['article', 'timestamp'], how='left')

# Based on API documentation and our analysis the missing views correspond to a day with 0 views
# Null rows therefore filled with 0
wiki_proc_df = wiki_proc_df.withColumn("views", F.when(wiki_proc_df.views.isNull(),0)
                                 .otherwise(wiki_proc_df.views))
# Dropping timestamp column
wiki_proc_df = wiki_proc_df.drop('timestamp')

# Saving processed data to csv for part 3
wiki_proc_df.repartition(1).orderBy('article','rn').write.mode("overwrite").csv('data/processed/wiki')
# Imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType
from pyspark.sql import Window

import pyspark.sql.functions as F
import datetime
import time
import numpy as np

# Start Spark
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

def linear_interpolation(df,id_col,order_col,value_col,num_dec):
    """
    Apply linear interpolation to dataframe to fill gaps. Backfill and forwardfill rows if no non-empty entry before/ after.
    
    NEED TO UPDATE PRICE COLUMN
    IF ALL EMPTY BEFORE BACKFILL/ AFTER FORWARDFILL/ IN BETWEEN INTERPOLATE
    
    CASE WHEN NULL THEN +1 WHEN NON NULL AGAIN THEN 0
    """
     
    # create row number over window and a column with row number only for non missing values
    w = Window.partitionBy('stock').orderBy('rn')
    new_df = df.withColumn('rn_not_null',F.when(F.col(value_col).isNotNull(),F.col('rn')))

    # find last non missing value
    w_before = Window.partitionBy(id_col).orderBy(order_col).rowsBetween(Window.unboundedPreceding,-1) #checking all previous rows
    new_df = new_df.withColumn('start_val',F.last(value_col,True).over(w_before)) # returning last non negative value
    new_df = new_df.withColumn('start_rn',F.last('rn_not_null',True).over(w_before)) # returning last non negative row number

    # find next non missing value
    w_after = Window.partitionBy(id_col).orderBy(order_col).rowsBetween(0,Window.unboundedFollowing) # checking all following rows
    new_df = new_df.withColumn('end_val',F.first(value_col,True).over(w_after)) # returning next non negative value
    new_df = new_df.withColumn('end_rn',F.first('rn_not_null',True).over(w_after)) # returning next non negative row number

    # create references to gap length and current gap position
    new_df = new_df.withColumn('diff_rn',F.col('end_rn')-F.col('start_rn')) # calculating number of empty rows between last non negative value and next non negative value
    new_df = new_df.withColumn('curr_rn',F.col('diff_rn')-(F.col('end_rn')-F.col('rn'))) # calculating number of empty rows between current row and next non negative value

    # calculate linear interpolation value
    lin_interp_func = (F.col('start_val')+((F.col('end_val')-F.col('start_val'))/F.col('diff_rn'))*F.col('curr_rn'))
    
    # fill blanks with linear interpolation value if non-empty value before and after
    # back-fill blanks with end_val if no non-empty value before
    # forward-fill blanks with start_val if no non-empty value after
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

# Filtering to S&P stocks
stocks_filtered_df = stocks_df.where(F.col('stock').contains('Nordamerika_USA-S-P'))
# Filtering to 1000 of these stocks
stocks_filtered_df = stocks_filtered_df.filter(F.col("stock").isin(stocks_filtered_df
                                                                   .select("stock").distinct().rdd.flatMap(lambda x:x).collect()[:1000]))

#Creating date dataframe with row number
date_df = spark.createDataFrame(((np.array([date_range(datetime.date(2020, 12, 31), enddate=True), range(-5,1007)])).T).tolist(), schema=["date", "rn"])

# Creating dataframe that is a combination of every date and stock
# Repartitioning to improve performance
stock_distinct = stocks_filtered_df.select('stock').distinct()
stock_date_df = date_df.repartition(20).crossJoin(stock_distinct)

# Joining to original stocks data (dropping volume variable)
# Using left join to identify dates missing price data
stocks_proc_df = stock_date_df.join(stocks_filtered_df.select('stock', 'date', 'price'), on=['stock', 'date'], how='left')

#Using 5 decimal places as this is the maximum precision used in our relevant stocks
stocks_proc_df = linear_interpolation(stocks_proc_df, 'stock', 'rn', 'price', 5)

# Filter to the 1000 days we are interested in (removing extra data used for interpolation)
stocks_proc_df = stocks_proc_df.filter((F.col("rn") > 0) & (F.col("rn") < 1001))
# Dropping date column
stocks_proc_df = stocks_proc_df.drop('date')

# Saving dataset
stocks_proc_df.repartition(1).orderBy('stock','rn').write.mode("overwrite").csv('data/processed/stock')

# Changing from storing each data point on row basis to as a list
stocks_proc_df = stocks_proc_df.withColumn("price_list", F.collect_list("price").over(Window.partitionBy("stock").orderBy("rn"))) \
    .filter(F.col("rn") == 1000).drop("rn", "price")
stocks_proc_df = stocks_proc_df.withColumn("price_list_string", F.col("price_list").cast(StringType())).drop("price_list")

# Saving processed data as parquet for part 1
stocks_proc_df.repartition(1).write.mode("overwrite").csv('data/processed/stock_list')


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
# Filtering to 1000 most important articles
wiki_df = wiki_df.filter(F.col("article").isin(wiki_df.select("article").distinct().rdd.flatMap(lambda x:x).collect()[:1000]))

# Filtering date_df to only 1000 dates we're interested in (interpolation not required for this dataset)
date_df = date_df.filter((F.col("rn") > 0) & (F.col("rn") < 1001))
# Renaming date column in date dataframe to allow join
wiki_date_df = date_df.withColumnRenamed("date","timestamp")

# Creating dataframe that is a combination of every date and article
article_distinct = wiki_df.select('article').distinct()
wiki_date_df = wiki_date_df.repartition(20).crossJoin(article_distinct)

# Joining to original wikipedia data
# Using left join to identify dates missing views data
wiki_proc_df = wiki_date_df.join(wiki_df, on=['article', 'timestamp'], how='left')

# Based on API documentation and our analysis the missing views correspond to a day with 0 views
# Null rows therefore filled with 0
wiki_proc_df = wiki_proc_df.withColumn("views", F.when(wiki_proc_df.views.isNull(),0)
                                 .otherwise(wiki_proc_df.views))
# Dropping timestamp column
wiki_proc_df = wiki_proc_df.drop('timestamp')

# Saving processed data to csv for part 3
wiki_proc_df.repartition(1).orderBy('article','rn').write.mode("overwrite").csv('data/processed/wiki')

# Changing from storing each data point on row basis to as a list
wiki_proc_df = wiki_proc_df.withColumn("views_list", F.collect_list("views").over(Window.partitionBy("article").orderBy("rn"))) \
    .filter(F.col("rn") == 1000).drop("rn", "views")
wiki_proc_df = wiki_proc_df.withColumn("views_list_string", F.col("views_list").cast(StringType())).drop("views_list")

# Saving processed data as parquet for part 1
wiki_proc_df.repartition(1).write.mode("overwrite").csv('data/processed/wiki_list')