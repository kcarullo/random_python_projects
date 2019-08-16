import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('automobile.csv')
df.head()
df.info()
# headers are missing adding headers
column_names = ['symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration',
                'num_of_doors', 'body_style', 'drive_wheels','drive_wheels',
                'engine_location','wheel_base', 'length','width','height',
                'curb_weight','num_of_cylinders', 'engine_type','fuel_system',
                'bore','stroke','compression_ratio', 'horsepower','peak_rpm',
                'city_mpg', 'highway_mpg','price']

# re run 
df = pd.read_csv('automobile.csv', header=-1, na_values='?',names=column_names)
df.head()
df.columns
df.fuel_type.unique()

# check data types 
df.dtypes

# check for missing values
df.isnull().any()

# filter columns with NaNs to find out specifically how many are missing per column
df[df.columns[df.isnull().any()].tolist()].isnull().sum()

# explore categorical data
df.make.unique() 
df.fuel_type.unique()
df.fuel_system.unique()

# Remove unused columns
df.drop(['make', 'symboling', 'normalized_losses'], axis = 1, inplace=True)
df.head()

# Dealing with missing data 
df.num_of_doors.isnull()
df[df.num_of_doors.isnull()] # the 2 NaN are sedans

# see how many sedans have 2 vs 4 doors
df.num_of_doors[df.body_style == 'sedan'].value_counts() # 79 sedans have 4 doors vs 15 with 2
df.loc[27, 'num_of_doors'] = 'four'
df.loc[63, 'num_of_doors'] = 'four'
df[df.columns[df.isnull().any()].tolist()].isnull().sum() # now showing 0 NaNs for number of doors

# bore column NaNs
df[df.bore.isnull()]
df.bore.fillna(df.bore.mean(), inplace=True)

# stroke column NaNs
df.stroke.fillna(df.stroke.mean(), inplace=True)

# horse power column NaNs
df.horsepower.fillna(df.horsepower.mean(), inplace=True)

# horse power column NaNs
df.peak_rpm.fillna(df.peak_rpm.mean(), inplace=True)

# price column NaNs
df.drop(df[df.price.isnull()].index, axis=0, inplace=True)

df.head()
df[df.columns[df.isnull().any()].tolist()].isnull().sum() # No missing data anymore







