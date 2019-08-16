import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('automobile.csv')
df.head()
df.info()
# headers are missing adding headers
column_names = ['symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration', 'num_of_doors', 
              'body_style', 'drive_wheels', 'engine_location', 'wheel_base', 'length', 'width', 'height', 
              'curb_weight', 'engine_type', 'num_of_cylinders', 'engine_size', 'fuel_system', 'bore', 
              'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price']

# re run 
df = pd.read_csv('automobile.csv', header=-1, names=column_names,na_values='?')
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
cat_columns = ['fuel_type', 'fuel_system', 'aspiration', 'num_of_doors', 
               'body_style', 'drive_wheels', 'engine_location', 'engine_type']
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

### Dealing with categorical columns
# num_of_cylinders columns
df.num_of_cylinders.value_counts()
df.loc[df.index[df.num_of_cylinders == 'four'], 'num_of_cylinders'] = 4
df.loc[df.index[df.num_of_cylinders == 'six'], 'num_of_cylinders'] = 6
df.loc[df.index[df.num_of_cylinders == 'five'], 'num_of_cylinders'] = 5
df.loc[df.index[df.num_of_cylinders == 'eight'], 'num_of_cylinders'] = 8
df.loc[df.index[df.num_of_cylinders == 'two'], 'num_of_cylinders'] = 2
df.loc[df.index[df.num_of_cylinders == 'twelve'], 'num_of_cylinders'] = 12
df.loc[df.index[df.num_of_cylinders == 'three'], 'num_of_cylinders'] = 3
df.num_of_cylinders = df.num_of_cylinders.astype('int')

#
cat_columns = ['fuel_type', 'fuel_system', 'aspiration', 'num_of_doors', 
               'body_style', 'drive_wheels', 'engine_location', 'engine_type']
df = pd.get_dummies(df, columns = cat_columns, drop_first=True)
df.head()
# split data into train(80%) and test (20%) 
train, test = train_test_split(df, test_size=0.2, random_state = 42)
train # 164 row x 68 columns 
test # 41 rows x 68 columns 

# separate the dependent variable, price, from the training data
Y_train = train.price
X_train = train.drop(['price'], axis = 1)

# separate the dependent variable, price, from the test data
Y_test = train.price
X_test = train.drop(['price'], axis = 1)

### Train and execute the model
# create lin reg object 
lrm = linear_model.LinearRegression()

# Train the model using the training sets
lrm.fit(X_train, Y_train)

# make predictions using the test set
predicted_price = lrm.predict(X_test)

# Access the performance of the model 
r_square = r2_score(Y_test, predicted_price)
print(r_square)

# 
actual_data = np.array(Y_test)
for i in range(len(predicted_price)):
    actual = actual_data[i]
    predicted = predicted_price[i]
    explained = ((actual_data[i] - predicted_price[i])/actual_data[i])*100
    
    print('Actual Value ${:,.2f}, Predicted Value ${:,.2f} (%{:.2f})'.format(actual,
          predicted, explained))









