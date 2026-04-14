import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr

wr.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

df = pd.read_csv('data/raw/sales_data.csv', sep=';', engine='python')
print(df.head())

print(df.shape)

print(df.info())

print(df.describe().T)

list = df.columns.tolist()
print("All features listed \n",list)

print("Null values per column \n",df.isnull().sum())

duplicated_per_column =df.apply(lambda col: col.duplicated().sum())
print("Duplicated per column \n",duplicated_per_column)