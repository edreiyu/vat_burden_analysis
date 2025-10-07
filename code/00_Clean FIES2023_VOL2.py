import pandas as pd
import numpy as np
import os
import polars as pl

# chunk_size = 50000
# chunks = pd.read_csv('file.csv', chunksize=chunk_size)

# Set working directory
os.chdir('/Users/edreiyu/Work-Offline/')
print(os.getcwd()) #get current working directory

# Import FIES volume 2 data
hh_vol2 = pd.read_csv('FIES PUF 2023_HH Summary and Volume 2 merged.csv')

# Filter columns to exclude cash, kind, and gifts columns
exclude_keywords = ['CASH', 'KIND', 'GIFTS']
hh_vol2_totals = hh_vol2.loc[:, ~hh_vol2.columns.str.contains('CASH|KIND|GIFTS', na=False)]

print(hh_vol2_totals.columns.tolist())
print(hh_vol2_totals.shape[1]) #print number of [1] columns in df

# Import HH summary data with cash abroad and domestic columns
hh_cash_abroad = pd.read_csv('FIES 2023 PUF VOL2/FIES PUF 2023 Volume2 Household Summary.CSV')

# merge cash domestic and abroad columns
hh_vol2_totals_complete = pd.merge(hh_vol2_totals, hh_cash_abroad[['SEQ_NO', 'CASH_ABROAD', 'CASH_DOMESTIC']], on='SEQ_NO', how='left')

# Remove 2 extra columns on region and province
hh_vol2_totals_complete = hh_vol2_totals_complete.drop(columns=['W_REGN.y', 'W_PROV.y'])

# Rename columns to remove .x suffix
hh_vol2_totals_complete = hh_vol2_totals_complete.rename(columns={'W_REGN.x': 'W_REGN', 'W_PROV.x': 'W_PROV'})

# Check if there are any missing values in the merged cash columns
if hh_vol2_totals_complete['CASH_ABROAD'].isnull().any() or hh_vol2_totals_complete['CASH_DOMESTIC'].isnull().any():
    print("There are missing values in the cash columns.")

# Save the final dataframe to a new CSV file
hh_vol2_totals_complete.to_csv('FIES2023_VOL2_COMPLETE.csv', index=False)