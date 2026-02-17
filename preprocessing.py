import pandas as pd
from pathlib import Path


def load_data():
    path = "data/sales.csv"
    
    df = pd.read_csv(path)
    
    # Convert date
    df["Factuurdatum"] = pd.to_datetime(df["Factuurdatum"], dayfirst=True)
    
    # Fix Liter column (replace comma decimal if needed)
    df["Liter"] = (
        df["Liter"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    
    df["Liter"] = pd.to_numeric(df["Liter"], errors="coerce")
    
    print("Data loaded successfully.")
    print("Shape:", df.shape)
    print(df.info())
    
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.columns)
