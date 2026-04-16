import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr

wr.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

df = pd.read_csv('data/raw/sales_data.csv', sep=';', engine='python')
print(df.head())

buffer = io.StringIO()
print(df.info(buf=buffer))
s = buffer.getvalue()

duplicated_per_column =df.apply(lambda col: col.duplicated().sum())
print("Duplicated per column \n", duplicated_per_column)

plt.figure(figsize=(15, 10))

sns.heatmap(df.corr(numeric_only=True,), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)

plt.title('Correlation Heatmap')
plt.show()

with open("exploratorydata.txt", "w") as f:
  f.write(str("Number of rows and columns, respectively \n"))
  f.write(str(df.shape)) 
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("Data types of each column \n"))
  f.write(str(s))
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("Descriptive statistics \n"))
  f.write(str(df.describe()))
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("Null values per column \n"))
  f.write(str(df.isnull().sum()))
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("All features listed \n"))
  f.write(str(list))
  f.write(str("\n"))
  f.write(str("Duplicated per column \n"))
  f.write(str(duplicated_per_column))