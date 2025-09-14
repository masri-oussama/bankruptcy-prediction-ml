import pandas as pd
import numpy as np
import seaborn as sns

##EDA

## Load Data
data = pd.read_csv('/Users/oussamaelmasri/Documents/MLP/Original data.csv')
## Display first few rows of the dataset
data.head()
## Display the last few rows of the data set
data.tail()
## Display the shape of the dataset
data.shape
## Display information about the dataset
data.info()
## Check for missing values
data.isnull().sum()
## Display summary statistics
data.describe()
## Check for duplicate rows
data.duplicated().sum()
## Display data types of each column
data.dtypes
## Display number of unique values in each column
data.nunique()
## Display value counts for the target variable
data['Bankrupt?'].value_counts()