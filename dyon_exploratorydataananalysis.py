from altair.datasets import data
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr

wr.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

NAME_MAP = {
    'Aantal_x': 'S_Aantal',
    'Aantal_y': 'V_Aantal',
    'Factuurdatum': 'Datum',
    'Factuurnummer': 'S_Factuurnummer',
    'Naam': 'S_Naam',
    'Identificatienummer': 'S_Identificatienummer',
    'Commerciële naam': 'S_Commerciële_naam',
    'Bedrijfscategorieën': 'S_Bedrijfscategorieën',
    'Land': 'S_Land',
    'Vertegenwoordiger': 'S_Vertegenwoordiger',
    'Stad': 'S_Stad',
    'Prijsklasse': 'S_Prijsklasse',
    'Naam product': 'S_Naam_product', 
    'Artikelnummer': 'S_Artikelnummer',
    'Productcategorieën': 'S_Productcategorieën',
    'Grondstof': 'S_Grondstof',
    'Liter': 'S_Liter',
    'Event_Name': 'E_Event_Name',
    'Product': 'V_Product',
    'Voorraadlocatie': 'V_Voorraadlocatie',
    'Bedrijf': 'V_Bedrijf',
    'Transactienummer': 'V_Transactienummer',
    'Type': 'V_Type'
    }

FEATURE_MAP = [
  'S_Factuurnummer',
  'Datum',
  'S_Naam',
  'S_Land',
  'S_Vertegenwoordiger',
  'S_Stad', 
  'S_Naam_product',
  'S_Productcategorieën',
  'S_Grondstof',
  'S_Liter',
  'E_Event_Name',
  'V_Product',
  'V_Voorraadlocatie',
  'V_Bedrijf',
  'V_Type',
  'V_Aantal'
]

BEER_NAME_MAP = {
    "Hoop US Lager": "Hoop Lager",
    "Hoop Lager": "Hoop Lager",
    "Hoop Kaper Tropical - Summer Session IPA": "Hoop Kaper Tropical IPA",
}

CORE_BEERS = [
    "Hoop Bleke Nelis",
    "Hoop Lager",
    "Hoop Kaper Tropical IPA",
]


sales_database = pd.read_csv('data/raw/sales_data.csv', sep=';', engine='python')
events_database = pd.read_csv('data/events.csv', sep=',', engine='python')
voorraad1_database = pd.read_csv('data/Voorraad-2026-02-16-01.23.46.007.csv', sep=';', engine='python')

sales_database['Factuurdatum'] = pd.to_datetime(sales_database['Factuurdatum'], dayfirst=True)
voorraad1_database['Datum'] = pd.to_datetime(voorraad1_database['Datum'], dayfirst=True)


sales_database['day_temp'] = sales_database['Factuurdatum'].dt.day
sales_database['month_temp'] = sales_database['Factuurdatum'].dt.month
voorraad1_database['day_temp'] = voorraad1_database['Datum'].dt.day
voorraad1_database['month_temp'] = voorraad1_database['Datum'].dt.month 


database = pd.merge(
  sales_database,
  events_database,
  left_on=['day_temp', 'month_temp'],
  right_on=['day', 'month'],
  how='left'
).merge(voorraad1_database, left_on=['day_temp', 'month_temp', 'Naam product'], right_on=['day_temp', 'month_temp', 'Product'], how='left')


database.drop(columns=['day_temp', 'month_temp', 'day', 'month', 'Datum'], inplace=True)
database.rename(columns= NAME_MAP, inplace=True)
#database["S_Grondstof"] = database["S_Grondstof"].replace(BEER_NAME_MAP)

#def filter_core_beers(database):
  #  print("Filtering core beers...")
    #return database[database["S_Grondstof"].isin(CORE_BEERS)]


df = database[FEATURE_MAP]


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
  f.write(str("Number of rows and columns, respectively: \n"))
  f.write(str(df.shape)) 
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("Data types of each column: \n"))
  f.write(str(s))
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("Descriptive statistics: \n"))
  f.write(str(df.describe()))
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("Null values per column: \n"))
  f.write(str(df.isnull().sum()))
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("All features listed: \n"))
  f.write(str(list))
  f.write(str("\n"))
  f.write(str("Duplicated per column: \n"))
  f.write(str(duplicated_per_column))