from altair.datasets import data
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr

wr.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

# Define the mapping of original column names to new column names
NAME_MAP = {
    'Aantal_x': 'S_Aantal',
    'Aantal_y': 'V_Aantal',
    'Factuurdatum': 'Datum',
    'Factuurnummer': 'S_Factuurnummer',
    'Naam': 'S_Naam',
    'Identificatienummer': 'S_Identificatienummer',
    'Commerciële naam': 'S_Commerciële_naam',
    'Bedrijfscategorieën': 'S_Bedrijfscategorieen',
    'Land': 'S_Land',
    'Vertegenwoordiger': 'S_Vertegenwoordiger',
    'Stad': 'S_Stad',
    'Prijsklasse': 'S_Prijsklasse',
    'Naam product': 'S_Naam_product', 
    'Artikelnummer': 'S_Artikelnummer',
    'Productcategorieën': 'S_Productcategorieen',
    'Grondstof': 'S_Grondstof',
    'Liter': 'S_Liter',
    'Event_Name': 'E_Event_Name',
    'Product': 'V_Product',
    'Voorraadlocatie': 'V_Voorraadlocatie',
    'Bedrijf': 'V_Bedrijf',
    'Transactienummer': 'V_Transactienummer',
    'Type': 'V_Type'
    }

# Define the list of features to keep in the final dataframe
FEATURE_MAP = [
  'S_Factuurnummer',
  'Datum',
  'S_Naam',
  'S_Land',
  'S_Vertegenwoordiger',
  'S_Stad', 
  'S_Naam_product',
  'S_Productcategorieen',
  'S_Grondstof',
  'S_Liter',
  'E_Event_Name',
  'V_Product',
  'V_Voorraadlocatie',
  'V_Bedrijf',
  'V_Type',
  'V_Aantal',
  'S_Container',
]

# Define a mapping for beer names to standardize them
BEER_NAME_MAP = {
    "Hoop US Lager": "Hoop Lager",
    "Hoop Lager": "Hoop Lager",
    "Hoop Kaper Tropical - Summer Session IPA": "Hoop Kaper Tropical IPA",
}

# Define the list of core beers to filter the dataset
CORE_BEERS = [
    "Hoop Bleke Nelis",
    "Hoop Lager",
    "Hoop Kaper Tropical IPA",
]

#LOAD THE DATA
sales_database = pd.read_csv('data/raw/sales_data.csv', sep=';', engine='python', decimal = ',')
events_database = pd.read_csv('data/events.csv', sep=',', engine='python', decimal = ',')
voorraad1_database = pd.read_csv('data/Voorraad-2026-02-16-01.23.46.007.csv', sep=';', engine='python', decimal = ',')

#PREPARE THE DATA BY EXTRACTING DAY AND MONTH FROM THE DATE COLUMNS
sales_database['Factuurdatum'] = pd.to_datetime(sales_database['Factuurdatum'], dayfirst=True)
sales_database['day_temp'] = sales_database['Factuurdatum'].dt.day
sales_database['month_temp'] = sales_database['Factuurdatum'].dt.month

voorraad1_database['Datum'] = pd.to_datetime(voorraad1_database['Datum'], dayfirst=True)
voorraad1_database['day_temp'] = voorraad1_database['Datum'].dt.day
voorraad1_database['month_temp'] = voorraad1_database['Datum'].dt.month 

#MERGE THE DATAFRAMES ON THE EXTRACTED DAY AND MONTH COLUMNS BY AGGREGATING THE VOORRAAD DATA TO MATCH THE SALES DATA
voorraad_grouped = voorraad1_database.groupby(['day_temp', 'month_temp', 'Product']).agg({
    'Aantal': 'sum',           
    'Voorraadlocatie': 'first', 
    'Bedrijf': 'first',
    'Type': 'first'
}).reset_index()

#MERGE THE SALES, EVENTS, AND VOORRAAD DATAFRAMES TO CREATE A COMPREHENSIVE DATABASE FOR ANALYSIS
database = pd.merge(
  sales_database,
  events_database,
  left_on=['day_temp', 'month_temp'],
  right_on=['day', 'month'],
  how='left'
).merge(voorraad_grouped, left_on=['day_temp', 'month_temp', 'Naam product'], right_on=['day_temp', 'month_temp', 'Product'], how='left')

#CLEAN THE DATA BY DROPPING UNNECESSARY COLUMNS, RENAMING COLUMNS, STANDARDIZING BEER NAMES, AND FILLING MISSING VALUES IN THE EVENT NAME COLUMN
database.drop(columns=['day_temp', 'month_temp', 'day', 'month'], inplace=True, errors='ignore')
database.rename(columns= NAME_MAP, inplace=True)
database["S_Grondstof"] = database["S_Grondstof"].replace(BEER_NAME_MAP)
database['E_Event_Name'] = database['E_Event_Name'].fillna('None')

#EXTRACT THE CONTAINER TYPE FROM THE PRODUCT NAME AND CREATE A NEW COLUMN FOR IT
def extract_container(name):
    name = str(name).lower()

    if "20l" in name:
        return "Keg 20L"
    elif "50l" in name:
        return "Keg 50L"
    elif "0,75" in name or "75cl"in name:
        return "Bottle 75cl"
    elif "0,33" in name or "33cl" in name:
          if "blik" in name or "can" in name:
              return "Can 33cl"
          return "Bottle 33cl"
    elif "cadeau" in name or "set" in name or "pack" in name:
        return "Giftset"
    else:
        return "Other"

database["S_Container"] = database["S_Naam_product"].apply(extract_container)

#FILTER THE DATAFRAME TO INCLUDE ONLY THE CORE BEERS FOR FURTHER ANALYSIS
def filter_core_beers(database):
    print("Filtering core beers...")
    return database[database["S_Grondstof"].isin(CORE_BEERS)]

#Filter the dataframe to include only the necessary features, as well as only the core beers for further analysis
df = database[FEATURE_MAP]
df = filter_core_beers(df)

df["week"] = df["Datum"] - pd.to_timedelta(df["Datum"].dt.weekday, unit='D')
df["week"] = df["week"].dt.normalize()

# 1. AGGREGATE (This is where the columns are BORN)
weekly = (
  df.groupby(["week", "S_Grondstof", "S_Container"])
  .agg(
        S_Liter=("S_Liter", "sum"),
        V_Aantal=("V_Aantal", "sum"),
        # Numbers (Counts)
        S_Factuurnummer=("S_Factuurnummer", "nunique"),
        S_Naam=("S_Naam", "nunique"),
        S_Land=("S_Land", "nunique"),
        S_Stad=("S_Stad", "nunique"),
        S_Vertegenwoordiger=("S_Vertegenwoordiger", "nunique"),
        S_Productcategorieen=("S_Productcategorieen", "nunique"),
        V_Product=("V_Product", "nunique"),
        V_Bedrijf=("V_Bedrijf", "nunique"),
        # Names (The actual text)
        S_Land_Naam=("S_Land", lambda x: m.iloc[0] if not (m := x.mode()).empty else "None"),
        S_Klant_Naam=("S_Naam", lambda x: m.iloc[0] if not (m := x.mode()).empty else "None"),
        S_Vertegenwoordiger_Naam=("S_Vertegenwoordiger", lambda x: m.iloc[0] if not (m := x.mode()).empty else "None"),
        S_Stad_Naam=("S_Stad", lambda x: m.iloc[0] if not (m := x.mode()).empty else "None"),
        S_Productcategorieen_Naam=("S_Productcategorieen", lambda x: m.iloc[0] if not (m := x.mode()).empty else "None"),
        V_Product_Naam=("V_Product", lambda x: m.iloc[0] if not (m := x.mode()).empty else "None"),
        V_Bedrijf_Naam=("V_Bedrijf", lambda x: m.iloc[0] if not (m := x.mode()).empty else "None"),
        # Info
        E_Event_Name=("E_Event_Name", "first"),
        V_Voorraadlocatie=("V_Voorraadlocatie", "first"),
        V_Type=("V_Type", "first")
    )
  .reset_index()
)

# 2. GAP FILLING
full_range = pd.date_range(weekly["week"].min(), weekly["week"].max(), freq="W-MON")
beers = weekly["S_Grondstof"].unique()
containers = weekly["S_Container"].unique()
full_index = pd.MultiIndex.from_product([full_range, beers, containers], names=["week", "S_Grondstof", "S_Container"])

# Overwrite df with the gap-filled weekly data
df = weekly.set_index(["week", "S_Grondstof", "S_Container"]).reindex(full_index, fill_value=0).reset_index()

# 3. CLEANUP (Tell Python these are TEXT columns, not numbers)
# We include the new '_Naam' columns here!
text_cols = ['E_Event_Name', 'S_Land_Naam', 'S_Klant_Naam', 'S_Stad_Naam', 'S_Vertegenwoordiger_Naam', 'S_Productcategorieen_Naam', 'V_Product_Naam', 'V_Bedrijf_Naam', 'V_Voorraadlocatie', 'V_Type']
for col in text_cols:
    df[col] = df[col].replace(0, 'None')


# Create a separate dataframe for returns just to see what they are
returns = df[df['S_Liter'] < 0]

# For the rest of the EDA, focus on positive sales
df = df[df['S_Liter'] >= 0]

# Perform exploratory data analysis (EDA) on the cleaned and filtered dataframe
print(df.head())

# Get the number of rows and columns in the dataframe
buffer = io.StringIO()
print(df.info(buf=buffer, verbose=True, show_counts=True))
s = buffer.getvalue()

#Get duplicated values per column
duplicated_per_column =df.apply(lambda col: col.duplicated().sum())

# Force display settings for the file writing
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False) 
pd.set_option('display.max_colwidth', None)

#Overwrite the exploratory data analysis results to a text file
with open("exploratorydata.txt", "w") as f:
  f.write(str("Number of rows and columns, respectively2: \n"))
  f.write(str(df.shape)) 
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("Data types of each column: \n"))
  f.write(str(s))
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("Descriptive statistics: \n"))
  f.write(str(df.describe(include='all')))
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("Null values per column: \n"))
  f.write(str(df.isnull().sum()))
  f.write(str("\n"))
  f.write(str("\n"))
  f.write(str("All features listed: \n"))
  f.write(str(df.columns.tolist()))
  f.write(str("\n"))
  f.write(str("Duplicated per column: \n"))
  f.write(str(duplicated_per_column))
  f.write(str(df.dropna().head(10)))
  f.write(str("\n"))
  f.write(str(f"Total rows: {len(df)}"))
  f.write(str("\n"))
  f.write(str((f"Number of return entries: {len(returns)}")))

# Create a correlation heatmap to visualize the relationships between numeric features in the dataframe
plt.figure(figsize=(15, 10))

sns.heatmap(df.corr(numeric_only=True,), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)

plt.title('Correlation Heatmap')
plt.show()

print(df['S_Container'].value_counts())