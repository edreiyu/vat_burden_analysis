
import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Configuration
VAT_RATE = 0.12

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pl.Config.set_float_precision(2)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_width_chars(1000)

# LOAD DATA

fies_raw = pl.read_parquet('clean_data/fies_raw_vat_etr.parquet')
exempt_list = pl.read_csv('clean_data/vat_exempt_list.csv')

# HELPER

def clean_item_code(code):
    """Remove first 6 characters and add leading zero if exactly 6 chars remain."""
    cleaned = code[6:]
    return "0" + cleaned if len(cleaned) == 6 else cleaned

def extract_major_code(code):
    """Extract first 2 characters for major category code."""
    return code[:2]

def get_expenditure_items(df, major_code):
    """Get list of expenditure item codes for a given major category."""
    return (
        df
        .filter(pl.col('item_code_major') == major_code)
        .select('new_item_code')
        .to_series()
        .to_list()
    )

# CLEAN AND STANDARDIZE ITEM CODES

# Clean exempt_list item codes - Add leading zero for 6-digit item codes; then get first 2 digits to classify major categories
exempt_list = exempt_list.with_columns([
    pl.col('item_code')
      .cast(pl.Utf8)
      .map_elements(clean_item_code, return_dtype=pl.Utf8)
      .alias('new_item_code'),

    pl.col('item_code')
      .cast(pl.Utf8)
      .map_elements(clean_item_code, return_dtype=pl.Utf8)
      .map_elements(extract_major_code, return_dtype=pl.Utf8)
      .alias('item_code_major')
])

# CREATE VATABLE AND EXEMPT ITEM LISTS

vatable_items = (
    exempt_list
    .filter(pl.col('vatable') == 'vatable')
    .select('new_item_code')
    .to_series()
    .to_list()
)

exempt_items = (
    exempt_list
    .filter(pl.col('vatable') == 'exempt')
    .select('new_item_code')
    .to_series()
    .to_list()
)

# CREATE MAJOR EXPENDITURE CATEGORY GROUPINGS

category_codes = {
    'food': '01',
    'alcohol_&cigarettes': '02',
    'clothing': '03',
    'housing_&utilities': '04',
    'furnishings_&maintenance': '05',
    'health': '06',
    'transport': '07',
    'ict_&equipments': '08',
    'recreation': '09',
    'education': '10',
    'resto_&accomodation': '11',
    'insurance_&financial': '12',
    'personal_care': '13',
    'family_occasions': '15'
    #'other_disbursements': '17'
}

expenditure_categories = {
    name: get_expenditure_items(exempt_list, code)
    for name, code in category_codes.items()
}


# =============================================================================
# SELECT ONLY COLUMNS FOR EDUCATION EXPENDITURES: category_code == 10
# =============================================================================

# 1) Get education item codes (expenditure_category == 10)
education_codes = (
    exempt_list
    .filter(pl.col("item_code_major") == "10")   # or pl.col("expenditure_category") == 10 if that column exists
    .select("new_item_code")
    .to_series()
    .to_list()
)

# 2) Keep only those columns from fies_raw (plus IDs if needed)
id_cols = ["W_REGN", "W_PROV", "SEQ_NO", "FSIZE", "RPSU", "URB", "NPCINC", "RFACT", "MEM_RFACT",
            "TOINC", "TOTEX", "TOTDIS", "OTHREC", "TOREC", "PERCAPITA", "LC101_LNO", "LC03_REL", 
            "LC04_SEX", "LC05_AGE", "LC05B_ETHNICITY", "LC06_MSTAT", "LC07_HGC_LEVEL", "LC07_GRADE", 
            "NEWEMPSTAT", "PWGTPRV", "senior_count", 'total_vatable_expenditures', 'total_exempt_expenditures', 
            'total_expenditures', 'estimated_vat_perHH', 'estimated_vat_foregone', 'vat_paid_to_total_expenditures', 
            'vat_foregone_to_total_expenditures']
education_df = fies_raw.select([c for c in id_cols + education_codes if c in fies_raw.columns])

# all education codes present in education_df (exclude id cols)
edu_cols = [c for c in education_codes if c in education_df.columns]
# subset codes of specific goods/services
target_codes = ["1010121", "1010141", "1010144", "1010161", "1010181", "1010184", "1010261", "1010241", "1010244", "1020121", "1020144", "1020341", "1020521", "1030021", "1030041", "1030043", "1040021", "1040041", "1040061", "1040081", "1010123", "1010143", "1010165", "1010183", "1010185", "1010223", "1010243", "1010245", "1020123", "1020145", "1020323", "1020523", "1090100"]
target_cols = [c for c in target_codes if c in education_df.columns]

education_df_sum = education_df.with_columns([
    pl.sum_horizontal(edu_cols).alias("all_education_total"),
    pl.sum_horizontal(target_cols).alias("private_education_total"),
])

education_df_sum.write_csv('outputs/education_df_sum.csv')
