"""
VAT Burden Analysis - Data Processing Script
==============================================

This script cleans and processes FIES 2023 data for analyzing VAT burden by income decile.

Main objectives:
1. Categorize expenditure items as vatable or exempt
2. Create major expenditure groupings (food, health, education, etc.)
3. Calculate budget shares by income group
4. Estimate VAT paid and foregone by income group
5. Calculate effective VAT rates by decile
"""

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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_item_code(code):
    """Remove first 6 characters and add leading zero if exactly 6 chars remain."""
    cleaned = code[6:]
    return "0" + cleaned if len(cleaned) == 6 else cleaned


def extract_major_code(code):
    """Extract first 2 characters for major category code."""
    return code[:2]


def rename_column_name(col):
    """Remove TOTAL_ prefix and add leading zero if needed."""
    if col.startswith('TOTAL_'):
        code = col[6:]
        return "0" + code if len(code) == 6 else code
    return col


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

print("Loading data...")
exempt_list = pl.read_csv('clean_data/vat_exempt_list.csv')
fies_raw = pl.read_parquet('clean_data/FIES2023_VOL2_COMPLETE_MONTHLY_RFACT-ADJUSTED.parquet')
senior_counts = (
    pl.read_parquet('clean_data/senior_counts.parquet')
    .select([
        pl.exclude(['W_REGN', 'W_PROV'])
    ])
)
# Merge senior counts into FIES data
fies_raw = fies_raw.join(senior_counts, on='SEQ_NO', how='left')

print(f"FIES data shape: {fies_raw.shape}")
print(f"Exempt list shape: {exempt_list.shape}")
print(f"senior counts list shape: {senior_counts.shape}")

# =============================================================================
# CLEAN AND STANDARDIZE ITEM CODES
# =============================================================================

print("\nCleaning item codes...")

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

# Standardize FIES column names to match exempt_list
fies_raw = fies_raw.rename({col: rename_column_name(col) for col in fies_raw.columns})

print("Item codes cleaned and standardized")
print(exempt_list[['item_code_major', 'new_item_code', 'item_code', 'description', 'vatable']].head())


# =============================================================================
# CREATE VATABLE AND EXEMPT ITEM LISTS
# =============================================================================

print("\nCreating vatable and exempt item lists...")

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

print(f"Number of vatable items: {len(vatable_items)}")
print(f"Number of exempt items: {len(exempt_items)}")
print(f"Sample exempt codes: {exempt_items[:5]}")


# =============================================================================
# CREATE MAJOR EXPENDITURE CATEGORY GROUPINGS
# =============================================================================

print("\nCreating major expenditure categories...")

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

print(f"Created {len(expenditure_categories)} expenditure categories")


# =============================================================================
# CALCULATE TAX SHARES BY DECILE
# =============================================================================

print("\nCalculating tax shares by decile...")

tax_share = (
    fies_raw
    .with_columns([
        pl.sum_horizontal(vatable_items).alias('vatable_spending'),
        pl.sum_horizontal(exempt_items).alias('exempt_spending')
    ])
    .with_columns((pl.col('vatable_spending') + pl.col('exempt_spending')).alias('total_spending'))
    .group_by('NPCINC').agg([
        pl.sum('RFACT').alias('number_households'),
        pl.sum('TOINC').alias('total_income'),
        pl.sum('TOTEX').alias('total_HH_expenditures'),
        pl.sum('vatable_spending').alias('vatable_expenditures'),
        pl.sum('exempt_spending').alias('exempt_expenditures'),
        pl.sum('total_spending').alias('total_expenditures')
    ])
    .sort('NPCINC')
    .with_columns([
        (pl.col('vatable_expenditures') / pl.col('total_expenditures') * 100).alias('vatable_share_exp'),
        (pl.col('exempt_expenditures') / pl.col('total_expenditures') * 100).alias('exempt_share_exp'),
        (pl.col('vatable_expenditures') / pl.col('total_income') * 100).alias('vatable_share_income'),
        (pl.col('exempt_expenditures') / pl.col('total_income') * 100).alias('exempt_share_income'),
        (pl.col('vatable_expenditures') / pl.col('total_HH_expenditures') * 100).alias('vatable_share_HHexp'),
        (pl.col('exempt_expenditures') / pl.col('total_HH_expenditures') * 100).alias('exempt_share_HHexp')
    ])
)

print(tax_share)

# Calculate distribution across deciles
tax_share_bydecile = (
    tax_share
    .with_columns([
        (pl.col('vatable_expenditures') / pl.col('vatable_expenditures').sum() * 100)
        .alias('vatable_share_decile'),
        (pl.col('exempt_expenditures') / pl.col('exempt_expenditures').sum() * 100)
        .alias('exempt_share_decile')
    ])
)

print(tax_share_bydecile)


# =============================================================================
# CALCULATE BUDGET SHARES BY MAJOR CATEGORY
# =============================================================================

print("\nCalculating budget shares by major expenditure category...")

budget_share = (
    fies_raw
    .with_columns([
        pl.sum_horizontal(expenditure_categories[name]).alias(name)
        for name in expenditure_categories
    ])
    .group_by('NPCINC').agg(
        [pl.sum('RFACT').alias('number_households')] +
        [pl.sum(name) for name in expenditure_categories]
    )
    .sort('NPCINC')
    .with_columns(
        pl.sum_horizontal([pl.col(name) for name in expenditure_categories]).alias('total_expenditures')
    )
    .with_columns([
        (pl.col(category) / pl.col('total_expenditures') * 100).alias(f'{category}_share')
        for category in expenditure_categories
    ])
)

print(budget_share)


# =============================================================================
# CALCULATE VAT ESTIMATES AND EFFECTIVE TAX RATES
# =============================================================================

print("\nCalculating VAT estimates and effective tax rates...")

# Add VAT calculations to household-level data
fies_raw_vat_etr = (
    fies_raw
    .with_columns([
        pl.sum_horizontal(vatable_items).alias('total_vatable_expenditures'),
        pl.sum_horizontal(exempt_items).alias('total_exempt_expenditures')
    ])
    .with_columns([
        (pl.col('total_vatable_expenditures') + pl.col('total_exempt_expenditures')).alias('total_expenditures'),
        (pl.col('total_vatable_expenditures') * VAT_RATE).alias('estimated_vat_perHH'),
        (pl.col('total_exempt_expenditures') * VAT_RATE).alias('estimated_vat_foregone')
    ])
    .with_columns([
        ((pl.col('estimated_vat_perHH') / pl.col('total_expenditures')) * 100).alias('vat_paid_to_total_expenditures'),
        ((pl.col('estimated_vat_foregone') / pl.col('total_expenditures')) * 100).alias('vat_foregone_to_total_expenditures')
    ])
)

# Calculate aggregate VAT statistics by decile
main_summary = (
    fies_raw_vat_etr
    .group_by('NPCINC')
    .agg([
        pl.sum('RFACT').alias('number_households'),
        pl.sum('total_vatable_expenditures').alias('total_vatable_expenditures'),
        pl.sum('total_exempt_expenditures').alias('total_exempt_expenditures'),
        pl.sum('total_expenditures').alias('total_expenditures'),
        pl.sum('estimated_vat_perHH').alias('total_vat_paid'),
        pl.sum('estimated_vat_foregone').alias('total_vat_foregone'),
        # Mean VAT ETR by decile
        pl.mean('vat_paid_to_total_expenditures').alias('avg_vat_etr'),
        pl.mean('vat_foregone_to_total_expenditures').alias('avg_vat_foregone_etr')

        # Weighted sum for weighted mean calculation
        # (pl.col('vat_paid_to_total_expenditures') * pl.col('RFACT')).sum().alias('weighted_sum_vat_etr'),
        # (pl.col('vat_foregone_to_total_expenditures') * pl.col('RFACT')).sum().alias('weighted_sum_foregone_etr')
    ])
    .with_columns([
        # Population-level effective tax rates
        (pl.col('total_vat_paid') / pl.col('total_expenditures') * 100).alias('sum_vat_etr'),
        (pl.col('total_vat_foregone') / pl.col('total_expenditures') * 100).alias('sum_vat_foregone_etr')
    ])
    .sort('NPCINC')
)

print(main_summary)


# =============================================================================
# BREAKDOWN OF EXEMPT EXPENDITURES BY CATEGORY
# =============================================================================

print("\nCalculating exempt expenditure breakdown by category...")

# Identify categories with exempt items
categories_with_exempts = {
    name: [col for col in expenditure_categories[name] if col in exempt_items]
    for name in expenditure_categories.keys()
}
categories_with_exempts = {k: v for k, v in categories_with_exempts.items() if len(v) > 0}

print(f"Categories with exempt items: {list(categories_with_exempts.keys())}")

# Calculate exempt expenditures by category
exempt_by_category = (
    fies_raw_vat_etr
    .with_columns([
        pl.sum_horizontal(cols).alias(f'{name}_exempt')
        for name, cols in categories_with_exempts.items()
    ])
    .group_by('NPCINC')
    .agg([
        pl.sum('RFACT').alias('number_households'),
        pl.sum('total_vatable_expenditures').alias('total_vatable_expenditures'),
        pl.sum('total_exempt_expenditures').alias('total_exempt_expenditures'),
        pl.sum('total_expenditures').alias('total_expenditures')
    ] + [
        pl.sum(f'{name}_exempt').alias(f'{name}_exempt')
        for name in categories_with_exempts.keys()
    ])
    .sort('NPCINC')
    .with_columns([
        ((pl.col(f'{name}_exempt') / pl.col('total_expenditures')) * 100).alias(f'{name}_exempt_share')
        for name in categories_with_exempts.keys()
    ])
)

print(exempt_by_category)


# =============================================================================
# CALCULATE VAT PAID AND FOREGONE SUMMARY
# =============================================================================

print("\nCalculating VAT paid and foregone summary...")

vat_paid_and_foregone = (
    tax_share_bydecile
    .with_columns([
        (pl.col('vatable_expenditures') * VAT_RATE).alias('estimated_vat_paid'),
        (pl.col('exempt_expenditures') * VAT_RATE).alias('estimated_vat_foregone'),
        ((pl.col('vatable_expenditures') * VAT_RATE) / pl.col('total_expenditures') * 100).alias('effective_vat_rate'),
    ])
)

print(vat_paid_and_foregone)

# =============================================================================
# ADD ANALYSIS FOR HH WITH SENIOR CITIZENS
# =============================================================================

print("\nCalculating VAT burden for households with senior citizens...")

senior_summary = (
    fies_raw_vat_etr
    .filter(pl.col('senior_count') > 0)
    .group_by('NPCINC')
    .agg([
        pl.sum('RFACT').alias('number_households_with_seniors'),
        pl.sum('senior_count').alias('number_seniors'),
        # (pl.col('RFACT') * pl.col('senior_count')).alias('weighted_senior_count'),
        pl.sum('total_vatable_expenditures').alias('total_vatable_expenditures')
    ])
)
        # (pl.col('total_exempt_expenditures') * VAT_RATE).alias('estimated_vat_foregone')

print(senior_summary)
senior_summary.write_csv('outputs/senior_summary.csv')


# =============================================================================
# EXPORT RESULTS
# =============================================================================

print("\nExporting results...")

# Export main outputs
tax_share.write_csv('outputs/tax_share.csv')
tax_share_bydecile.write_csv('outputs/tax_share_bydecile.csv')
budget_share.write_csv('outputs/budget_share.csv')
main_summary.write_csv('outputs/main_summary.csv')
exempt_by_category.write_csv('outputs/exempt_by_category.csv')
vat_paid_and_foregone.write_csv('outputs/vat_paid_and_foregone.csv')

# Export processed household-level data
fies_raw_vat_etr.write_parquet('clean_data/fies_raw_vat_etr.parquet')

print("\nAll outputs exported successfully!")
print("\nOutput files:")
print("  - outputs/tax_share.csv")
print("  - outputs/tax_share_bydecile.csv")
print("  - outputs/budget_share.csv")
print("  - outputs/main_summary.csv")
print("  - outputs/exempt_by_category.csv")
print("  - outputs/vat_paid_and_foregone.csv")
print("  - clean_data/fies_raw_vat_etr.parquet")
print("\nData processing complete!")
