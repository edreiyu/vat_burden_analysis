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

# Set working directory
os.chdir('/Users/edreiyu/Documents/Work-Offline/FIES LFS Merge 2023')
print(os.getcwd()) #get current working directory

# ----------------------------------------------------------------------

#import HH members data
hh_members = pl.read_csv('FIES-LFS PUF 2023 Household Members.CSV')

columns = [
    'W_REGN', 'W_PROV', 'SEQ_NO', 'LC101_LNO', 'LC03_REL', 
    'LC04_SEX', 'LC05_AGE', 'LC05B_ETHNICITY',
    'LC06_MSTAT', 'LC07_HGC_LEVEL', 'LC07_GRADE', 'NEWEMPSTAT', 'PWGTPRV'
]

hh_members_subset = pl.read_csv('FIES-LFS PUF 2023 Household Members.CSV').select(columns)

# Count the number of senior citizens in each household (variable SEQ_NO)

# First, count seniors per household
senior_counts = (
    hh_members_subset
    .group_by('SEQ_NO')
    .agg(
        pl.col('LC05_AGE').filter(pl.col('LC05_AGE') >= 60).count().alias('senior_count')
    )
)

# Then join back to household heads
df_senior_counts = (
    hh_members_subset
    .filter(pl.col('LC03_REL') == 1)
    .join(senior_counts, on='SEQ_NO', how='left')
)

# Export df into parquet for mergin with the main FIES df
df_senior_counts.write_parquet('senior_counts.parquet')
