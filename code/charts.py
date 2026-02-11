"""
VAT Burden Analysis - Data Processing Script
==============================================
"""

import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pl.Config.set_float_precision(2)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_width_chars(1000)


def get_expenditure_items(df, major_code):
    """Get list of expenditure item codes for a given major category."""
    return (
        df
        .filter(pl.col('item_code_major') == major_code)
        .select('new_item_code')
        .to_series()
        .to_list()
    )


# =============================================================================
# LOAD DATA
# =============================================================================

exempt_list = pl.read_csv('clean_data/vat_exempt_list.csv')
fies_raw = pl.read_parquet('clean_data/fies_raw_vat_etr.parquet')
# fies_raw_selected = fies_raw.select([
#         pl.exclude(['W_REGN', 'W_PROV'])
#     ])

# Merge ??
# fies_raw = fies_raw.join(senior_counts, on='SEQ_NO', how='left')

print(f"FIES data shape: {fies_raw.shape}")
print(f"Exempt list shape: {exempt_list.shape}")
print(f"senior counts list shape: {senior_counts.shape}")

# =============================================================================
# Exempt spending by categories 
# =============================================================================