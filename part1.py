# Imports
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType

import pyspark.sql.functions as F
import datetime
import time
import numpy as np

#42.333091259002686 seconds with 5 time series for stocks dataset
#65.29546475410461 seconds with 1496 time series

start_time = time.time()

spark = SparkSession.builder.getOrCreate()

#INTERPOLATION FUNCTIONS
def date_range(date, step=1, enddate=False):
    """Return a list of 1000 dates from start date or before enddate."""
    date_range = []
    if enddate == True:
        date = date - datetime.timedelta(days=999)
    for i in range(1000):
        date_range.append(date)
        date = date + datetime.timedelta(days=step)
    return date_range

def fill_linear_interpolation(df,id_col,order_col,value_col,num_dec):
    """
    Apply linear interpolation to dataframe to fill gaps.
    Based on https://stackoverflow.com/questions/53077639/pyspark-interpolation-of-missing-values-in-pyspark-dataframe-observed
    
    Will also backfill and forwardfill as required.

    :param df: spark dataframe
    :param id_col: string or list of column names to partition by the window function
    :param order_col: column to use to order by the window function
    :param value_col: column to be filled

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
df = spark.read.csv("data/raw/MS1.txt", schema=schema, dateFormat="MM/dd/yyyy")

# Duplicate rows with identical data were identified so these are removed
df = df.distinct()


#Sampling data to USA P stocks (1491 stocks)
#stocks = ['39797.Nordamerika_USA-NYSE-Composite_Waste-Connections-Inc._WCN',
#          '42276.Nordamerika_USA-S-P100_Exxon-Mobil-Corp._XOM',
#          '32778.Nordamerika_USA-NASDAQ_Cohu-Inc._COHU',
#          '38410.Nordamerika_USA-NYSE-Composite_Embraer-Aircraft_ERJ',
#           '38206.Nordamerika_USA-NYSE-Composite_CNOOC-Ltd.-ADRs_CEO']
#df_filtered = df.where(F.col('stock').isin(stocks))

df_filtered = df.where(F.col('stock').contains('Nordamerika_USA-S-P'))

#Creating date dataframe with row number
date_df = spark.createDataFrame(((np.array([date_range(datetime.date(2020, 12, 31), enddate=True), range(1,1001)])).T).tolist(), schema=["date", "rn"])

# Creating dataframe that is a combination of every date and stock (can be optimized?)
stock_distinct = df_filtered.select('stock').distinct()
date_df = date_df.crossJoin(stock_distinct)

# Joining to original stocks data (dropping volume variable)
processed_data = date_df.join(df_filtered.select('stock', 'date', 'price'), on=['stock', 'date'], how='left')

#Using 5 decimal places as this is the maximum precision used in our relevant stocks
processed_data = fill_linear_interpolation(processed_data, 'stock', 'rn', 'price', 5)

#processed_data.write.mode("overwrite").csv('data/processed/sample_stock')
processed_data.write.mode("overwrite").csv('data/processed/stock')

stock_runtime = time.time()
print("Time to run stock dataset: %s seconds" % (stock_runtime - start_time))