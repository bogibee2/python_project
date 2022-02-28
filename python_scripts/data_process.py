

# 1.Load libraries #
import pandas as pd
import numpy as np


file_dir = "https://raw.githubusercontent.com/zhendata/Medium_Posts/master/City_Zhvi_1bedroom_2018_05.csv"

# read csv file into a Pandas dataframe
raw_df = pd.read_csv(file_dir)

# check first 5 rows of the file
# use raw_df.tail(5) to see last 5 rows of the file
raw_df.head(5)
raw_df[['RegionID']]

raw_df.loc[:,:]

df1 = raw_df.iloc[1:10,1:5]
X = raw_df.select_dtype['float']

y=raw_df.iloc[:,2].values