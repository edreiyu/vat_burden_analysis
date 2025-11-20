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

# Profiling the employment and income status of senior citizens

# pull-in the summary data to get the per capita income variable
hh_summary_subset = pl.read_csv('FIES-LFS PUF 2023 Household Summary.CSV').select(['SEQ_NO', 'NPCINC', 'FSIZE', 'TOINC', 'TOTEX', 'RFACT', 'MEM_RFACT'])

# join hh_members and hh_summary_subset
hh_members_summary = hh_members.join(hh_summary_subset, on='SEQ_NO', how='left')

# group seniors by ??
seniors_by_PCINC = (
    hh_members_summary
    .filter(pl.col('LC05_AGE') >= 60)
    .group_by('NEWEMPSTAT') # 'NPCINC'
    .agg(
        pl.sum('PWGTPRV').alias('PWGTPRV_sum'),
        pl.count().alias('count')
    )
)

# Total number of SCs is around 10.81 million in 2023
# EMPLOYED	1
# UNEMPLOYED	2
# NOT IN THE LABOR FORCE	3