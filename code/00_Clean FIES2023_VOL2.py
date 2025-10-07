import pandas as pd
import numpy as np
import os
import polars as pl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# Set global Polars display format
pl.Config.set_float_precision(2)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_width_chars(1000)

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



## Convert annual FIES into monthly by dividing by 12
fies_raw = pl.read_csv('clean_data/FIES2023_VOL2_COMPLETE.csv')

# Define non-amount columns
exclude_columns = [
    'W_REGN', 'W_PROV', 'SEQ_NO', 'RPROV', 'FSIZE', 'RPSU', 'RFACT', 
    'MEM_RFACT', 'URB', 'NPCINC', 'RPCINC', 'PRPCINC', 'PPCINC', 
    'RPCINC_NIR', 'W_REGN_NIR', 'LC101_LNO', 'LC03_REL', 'LC04_SEX', 
    'LC05_AGE', 'LC05A_5OVER', 'LC05B_ETHNICITY', 'LC06_MSTAT', 
    'LC07_HGC_LEVEL', 'LC07_GRADE', 'LC08_CURSCH', 'LC09_GRADTECH', 
    'LC09A_NFORMAL', 'LC10_CONWR', 'LC11_WORK', 'LC11A_ARRANGEMENT', 
    'LC12_JOB', 'LC12A_PROVMUN', 'LC14_PROCC', 'LC14_MAJOR_OCC', 
    'LC16_PKB', 'LC16_MAJOR_IND', 'LC17_NATEM', 'LC18_PNWHRS', 
    'LC19_PDAYS', 'LC19A_PHOURS', 'LC20_PWMORE', 'LC21_PLADDW', 
    'LC22_PFWRK', 'LC23_PCLASS', 'LC24_PBASIS', 'LC25_PBASIC', 
    'LC26_OJOB', 'LC27_NJOBS', 'LC28_THOURS', 'LC29_WWM48H', 
    'LC30_LOOKW', 'LC31_FLWRK', 'LC32_JOBSM', 'LC33_WEEKS', 
    'LC34_WYNOT', 'LC35_LTLOOKW', 'LC36_AVAIL', 'LC37_WILLING', 
    'LC38_PREVJOB', 'LC39_YEAR', 'LC39_MONTH', 'LC41_POCC', 
    'LC41_MAJOR_OCC', 'LC43_QKB', 'LC43_MAJOR_IND', 'NEWEMPSTAT', 
    'PWGTPRV'
]

# Get amount columns
all_columns = fies_raw.columns
amount_columns = [col for col in all_columns if col not in exclude_columns]

print(f"Converting {len(amount_columns)} amount columns from annual to monthly...")
print(f"Keeping {len(exclude_columns)} identifier/categorical columns unchanged")

# Show sample of columns being converted
print(f"\nSample amount columns being converted:")
for col in amount_columns[:10]:
    print(f"  - {col}")

# Convert to monthly amounts
fies_monthly = fies_raw.with_columns([
    (pl.col(col) / 12).alias(col) for col in amount_columns
])

# Save to CSV
fies_monthly.write_csv('clean_data/FIES2023_VOL2_COMPLETE_MONTHLY.csv')



# Calculate sum of RFACT column
rfact_sum = fies_monthly['RFACT'].sum()
mem_rfact_sum = fies_monthly['MEM_RFACT'].sum()
print(f"\nSum of RFACT column (number of households): {rfact_sum:.2f}")
print(f"\nSum of MEM_RFACT column (number of population): {mem_rfact_sum:.2f}")

# Calculate sum of RFACT column by NPINC
rfact_sum_by_npinc = (
    fies_monthly
    .group_by('NPCINC')
    .agg(pl.sum('RFACT').alias('RFACT_sum'))
    .sort('NPCINC')
)


## Multiply the monthly amounts  by RFACT to get household population totals
fies_monthly_x_rfact = fies_monthly.with_columns([
    (pl.col(col) * pl.col('RFACT')).alias(col) for col in amount_columns
])

# For population-weighted means
decile_analysis = (
    fies_monthly_x_rfact
    .group_by('NPCINC')
    .agg(pl.sum('TOREC').alias('total_torec')) # Population total
    .sort('NPCINC')
)
print(decile_analysis)

# Save to CSV
fies_monthly_x_rfact.write_csv('clean_data/FIES2023_VOL2_COMPLETE_MONTHLY_RFACT-ADJUSTED.csv')
# OR better - use Parquet (faster, smaller, preserves types)
fies_monthly_x_rfact.write_parquet('clean_data/FIES2023_VOL2_COMPLETE_MONTHLY_RFACT-ADJUSTED.parquet')

# Works with Polars: pl.read_parquet('file.parquet')
xx = pl.read_parquet('clean_data/FIES2023_VOL2_COMPLETE_MONTHLY_RFACT-ADJUSTED.parquet')


# Calculate sum of TOREC column for NPCINC == 1
torec_sum_npinc1 = fies_monthly_x_rfact.filter(pl.col('NPCINC') == 1)['TOREC'].sum()
print(f"\nSum of TOREC column for NPCINC == 1: {torec_sum_npinc1:.2f}")
