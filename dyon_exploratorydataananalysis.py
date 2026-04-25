from altair.datasets import data
import pandas as pd
import io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
import requests
import difflib

wr.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

SAVE_FILE = "exploratorydata.csv"
WEATHER_SAVE_FILE= "data\raw\weather_data.csv"

WEATHER_LAT = 52.3676
WEATHER_LON = 4.9041

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

#Note

# Define the list of features to keep in the final dataframe
FEATURE_MAP = [
  'S_Factuurnummer',
  'Datum',
  'S_Naam',
  'S_Land',
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

def standardize_names(series, threshold=0.8):
    """
    Cleans strings and groups similar names (e.g., Zaamdam -> Zaandam).
    """
    # basic cleanup
    clean_series = series.astype(str).str.strip().str.title().replace('Nan', np.nan)
    
    # Get unique non-null names
    unique_names = clean_series.dropna().unique().tolist()
    
    mapping = {}
    processed = set()

    for name in unique_names:
        if name in processed: continue
        # Find similar names in the list
        matches = difflib.get_close_matches(name, unique_names, n=10, cutoff=threshold)
        for match in matches:
            mapping[match] = name  # Map the typo to the first-seen version
            processed.add(match)
            
    return clean_series.map(mapping)

save_path = Path(SAVE_FILE)
save_weather_path = Path(WEATHER_SAVE_FILE)

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

# Standardize the 'Stad' and 'Naam' columns BEFORE grouping
sales_database['Stad'] = standardize_names(sales_database['Stad'])
sales_database['Naam'] = standardize_names(sales_database['Naam'])

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
database["S_Stad"] = database["S_Stad"].replace("Koog Ad Zaan", "Koog Aan De Zaan")
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

start_date = df['week'].min().strftime('%Y-%m-%d')
end_date = df['week'].max().strftime('%Y-%m-%d')
url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={WEATHER_LAT}"
        f"&longitude={WEATHER_LON}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&daily=temperature_2m_mean,precipitation_sum"
        "&timezone=Europe%2FBerlin"
    )

response = requests.get(url)

if response.status_code != 200:
    print("⚠ Weather API failed. Continuing without weather.")

data = response.json()

weather_df = pd.DataFrame({
    "date": pd.to_datetime(data["daily"]["time"]),
    "temp_mean": data["daily"]["temperature_2m_mean"],
    "rain_mm": data["daily"]["precipitation_sum"]
    })

#weather_df["week"] = weather_df["date"].dt.to_period("W").apply(lambda r: r.start_time)

#weekly_weather = (
    #weather_df.groupby("week")
    #.agg({
      # "temp_mean": "mean",
       # "rain_mm": "sum"
    #})
   # .reset_index()
#)

# 1. AGGREGATE (This is where the columns are BORN)
daily = (
    df.groupby(["Datum", "S_Grondstof", "S_Container"])
    .agg(
        S_Liter=("S_Liter", "sum"),
        V_Aantal=("V_Aantal", "sum"),
        # Reach Metrics (Unique counts renamed for clarity)
        S_Order_Count=("S_Factuurnummer", "nunique"),
        S_Customer_Count=("S_Naam", "nunique"),
        S_Country_Reach=("S_Land", "nunique"),
        S_City_Reach=("S_Stad", "nunique"),
        # Descriptive Labels (Most frequent name in that group)
        S_Main_City=("S_Stad", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        S_Main_Customer=("S_Naam", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        S_Main_Country=("S_Land", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        V_Product_Name=("V_Product", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        V_Business_Name=("V_Bedrijf", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        V_Storage_Location=("V_Voorraadlocatie", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        E_Event_Name=("E_Event_Name", "first"),
        V_Type=("V_Type", "first")
    )
    .reset_index()
)

# 2. GAP FILLING
full_range = pd.date_range(daily["Datum"].min(), daily["Datum"].max(), freq="D")
beers = daily["S_Grondstof"].unique()
containers = daily["S_Container"].unique()
full_index = pd.MultiIndex.from_product([full_range, beers, containers], names=["Datum", "S_Grondstof", "S_Container"])

# Overwrite df with the gap-filled weekly data
df = daily.set_index(["Datum", "S_Grondstof", "S_Container"]).reindex(full_index, fill_value=0).reset_index()
df = pd.merge(df, weather_df, left_on="Datum", right_on="date", how="left")

text_cols = ['E_Event_Name', 'S_Main_City', 'S_Main_Customer', 'S_Main_Country', 'V_Type', 'V_Product_Name', 'V_Business_Name', 'V_Storage_Location']
for col in text_cols:
    # Replace the '0' from reindexing with real empty values (NaN)
    df[col] = df[col].replace(0, np.nan).replace('None', np.nan)


# Create a separate dataframe for returns just to see what they are
returns = df[df['S_Liter'] < 0]

# For the rest of the EDA, focus on positive sales
df = df[df['S_Liter'] >= 0]
df = df.drop(columns=['date'], errors='ignore')

# Perform exploratory data analysis (EDA) on the cleaned and filtered dataframe
print(df.head())

# Get the number of rows and columns in the dataframe
buffer = io.StringIO()
print(df.info(buf=buffer, verbose=True, show_counts=True))
s = buffer.getvalue()

#Get duplicated values per column
duplicated_per_column =df.apply(lambda col: col.duplicated().sum())

print("Saving processed dataset...")

save_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(save_path, index=False)

# Force display settings for the file writing
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

desc = df.describe(include='all')
chunk_size = 3

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
  for i in range(0, len(desc.columns), chunk_size):
      # Slice the dataframe to get 3 columns at a time
    chunk = desc.iloc[:, i : i + chunk_size]
    f.write(chunk.to_string())
      
      # Add a visual separator and extra newlines between chunks
    f.write("\n\n" + "-"*80 + "\n\n")
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