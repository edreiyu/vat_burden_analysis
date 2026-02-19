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
fies_raw = pl.read_parquet('clean_data/FIES2023_VOL2_COMPLETE_MONTHLY_RFACT-ADJUSTED_RENT.parquet')
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
        pl.sum('TOTEX').alias('total_FIES_expenditures'),
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
        (pl.col('vatable_expenditures') / pl.col('total_FIES_expenditures') * 100).alias('vatable_share_HHexp'),
        (pl.col('exempt_expenditures') / pl.col('total_FIES_expenditures') * 100).alias('exempt_share_HHexp'),
        ])
    .with_columns([
        # Get the estimated VAT paid
        (pl.col('vatable_expenditures') * VAT_RATE).alias('estimated_vat_perHH'),
        (pl.col('exempt_expenditures') * VAT_RATE).alias('estimated_vat_foregone'),
        ])
    .with_columns([
        # Get ratio of VAT collection to INCOME, FIES EXPENDITURES, and TOTAL EXPENDITURES
        (pl.col('estimated_vat_perHH') / pl.col('total_income') * 100)
        .alias('vat_to_income'),
        (pl.col('estimated_vat_perHH') / pl.col('total_FIES_expenditures') * 100)
        .alias('vat_to_FIESexpenditures'),
        (pl.col('estimated_vat_perHH') / pl.col('total_expenditures') * 100)
        .alias('vat_to_TOTALexpenditures'),

        # Get ratio of EXEMPT EXPENDITURES to INCOME, FIES EXPENDITURES, and TOTAL EXPENDITURES
        (pl.col('estimated_vat_foregone') / pl.col('total_income') * 100)
        .alias('exempt_to_income'),
        (pl.col('estimated_vat_foregone') / pl.col('total_FIES_expenditures') * 100)
        .alias('exempt_to_FIESexpenditures'),
        (pl.col('estimated_vat_foregone') / pl.col('total_expenditures') * 100)
        .alias('exempt_to_TOTALexpenditures')
    ])    
)

print(tax_share)
# tax_share.write_csv('outputs/feb3/tax_share_with_ratios.csv')

# =============================================================================
#  Chart ETRs by decile
# =============================================================================

# Extract NumPy arrays directly from Polars (no pyarrow needed)
x  = tax_share["NPCINC"].to_numpy()
y1 = tax_share["vat_to_income"].to_numpy()
y2 = tax_share["vat_to_TOTALexpenditures"].to_numpy()

plt.figure(figsize=(8, 5))
plt.style.use("seaborn-v0_8-whitegrid")

# --- Remove vertical gridlines ---
plt.grid(axis="y", linestyle="--", linewidth=0.5)   # only horizontal
plt.gca().grid(axis="x", visible=False)             # remove vertical

# VAT/income (solid line)
plt.plot(x, y1, label="Share of VAT to Total Income", color="#5b84c4", linewidth=2.5)
# VAT/expenditure (dashed line)
plt.plot(x, y2, label="Share of VAT to Total Expenditures", color="#5b84c4", linewidth=2.5, linestyle="--")

# --- Black labels ---
plt.xlabel("Per Capita Income Decile", color="black")
plt.ylabel("In Percent", color = "black")
plt.xticks(np.arange(1, 11, 1), color="black")
plt.yticks(np.arange(0, 11, 2), color="black")
plt.ylim(0, 10)

# plt.title("Estimated VAT Burden of Households by Decile", color="black")
plt.legend()

# --- Add labels for first and last deciles (line annotations) ---
# For VAT/income
plt.text(x[0], y1[0] - 0.7, f"{y1[0]:.1f}%", color="black", fontsize=11)
plt.text(x[-1], y1[-1] - 0.7, f"{y1[-1]:.1f}%", color="black", fontsize=11)

# For VAT/expenditure
plt.text(x[0], y2[0] - 0.6, f"{y2[0]:.1f}%", color="black", fontsize=11)
plt.text(x[-1], y2[-1] - 0.7, f"{y2[-1]:.1f}%", color="black", fontsize=11)

plt.tight_layout()
plt.show()

# fig.savefig("vat_burdens_two_panels.png", dpi=300, bbox_inches="tight")

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
# CALCULATE COSTS OF EXEMPTING PRIVATE EDUCATION AND HEALTH BY DECILE
# =============================================================================

# educ_health = pl.read_csv('clean_data/private_educ_health.csv')
# educ_health_filtered = fies_raw.filter(pl.col('item_code').is_in(educ_health['item_code']))

private_educ_health = (
    exempt_list
    .filter(pl.col('new_item_code').is_in(
        ["1010121", "1010141", "1010144", "1010161", "1010181", "1010184", "1010221", "1010241", "1010244", "1020121", "1020141", "1020144", "1020321",
        "1020521", "1030021", "1030041", "1030043", "1040021", "1040041", "1040061", "1040081", "1010123", "1010143", "1010145", "1010163", "1010183", 
        "1010185", "1010223", "1010243", "1010245", "1020123", "1020143", "1020145", "1020323", "1020523", "0611122", "0611125", "0611126", "0621120",
        "0621929", "0623220", "0631020", "0632020", "0641010", "0641020"]))
    .select('new_item_code')
    .to_series()
    .to_list()
)

private_educ = (
    exempt_list
    .filter(pl.col('new_item_code').is_in(
        ["1010121", "1010141", "1010144", "1010161", "1010181", "1010184", "1010221", "1010241", "1010244", "1020121", "1020141", "1020144", "1020321",
        "1020521", "1030021", "1030041", "1030043", "1040021", "1040041", "1040061", "1040081", "1010123", "1010143", "1010145", "1010163", "1010183", 
        "1010185", "1010223", "1010243", "1010245", "1020123", "1020143", "1020145", "1020323", "1020523"]))
    .select('new_item_code')
    .to_series()
    .to_list()
)

medicines = (
    exempt_list
    .filter(pl.col('new_item_code').is_in(
        ["0611122", "0611125", "0611126"]))
    .select('new_item_code')
    .to_series()
    .to_list()
)

private_health = (
    exempt_list
    .filter(pl.col('new_item_code').is_in(
        ["0621120", "0621929", "0623220", "0631020", "0632020", "0641010", "0641020"]))
    .select('new_item_code')
    .to_series()
    .to_list()
)

exempt_educ_health = (
    fies_raw
    .with_columns([
        pl.sum_horizontal(vatable_items).alias('vatable_spending'),
        pl.sum_horizontal(exempt_items).alias('exempt_spending'),
        pl.sum_horizontal(private_educ).alias('private_educ_spending'),
        pl.sum_horizontal(medicines).alias('medicines_spending'),
        pl.sum_horizontal(private_health).alias('private_health_spending')
    ])
    .with_columns((pl.col('vatable_spending') + pl.col('exempt_spending')).alias('total_spending'))    
    .with_columns((pl.col('private_educ_spending') + pl.col('medicines_spending') + pl.col('private_health_spending')).alias('total_educ_health_spending'))
    .group_by('NPCINC').agg([
        pl.sum('RFACT').alias('number_households'),
        pl.sum('TOINC').alias('total_income'),
        pl.sum('TOTEX').alias('total_FIES_expenditures'),
        pl.sum('vatable_spending').alias('vatable_expenditures'),
        pl.sum('exempt_spending').alias('exempt_expenditures'),
        pl.sum('total_spending').alias('total_expenditures'),
        pl.sum('private_educ_spending').alias('private_educ_expenditures'),
        pl.sum('medicines_spending').alias('medicines_expenditures'),
        pl.sum('private_health_spending').alias('private_health_expenditures'),
        pl.sum('total_educ_health_spending').alias('total_educ_health_expenditures')
    ])
    .sort('NPCINC')
    .with_columns([
        (pl.col('private_educ_expenditures') / pl.col('total_expenditures') * 100).alias('private_educ_share_exp'),
        (pl.col('medicines_expenditures') / pl.col('total_expenditures') * 100).alias('medicines_share_exp'),
        (pl.col('private_health_expenditures') / pl.col('total_expenditures') * 100).alias('private_health_share_exp'),
        (pl.col('total_educ_health_expenditures') / pl.col('total_expenditures') * 100).alias('total_educ_health_share_exp'),
       
        (pl.col('private_educ_expenditures') / pl.col('total_income') * 100).alias('private_educ_share_income'),
        (pl.col('medicines_expenditures') / pl.col('total_income') * 100).alias('medicines_share_income'),
        (pl.col('private_health_expenditures') / pl.col('total_income') * 100).alias('private_health_share_income'),
        (pl.col('total_educ_health_expenditures') / pl.col('total_income') * 100).alias('total_educ_health_share_income'),


        (pl.col('private_educ_expenditures') / pl.col('total_FIES_expenditures') * 100).alias('private_educ_share_HHexp'),
        (pl.col('medicines_expenditures') / pl.col('total_FIES_expenditures') * 100).alias('medicines_share_HHexp'),
        (pl.col('private_health_expenditures') / pl.col('total_FIES_expenditures') * 100).alias('private_health_share_HHexp'),
        (pl.col('total_educ_health_expenditures') / pl.col('total_FIES_expenditures') * 100).alias('total_educ_health_share_HHexp'),

        ])
    .with_columns([
        # Get the estimated VAT paid
        (pl.col('private_educ_expenditures') * VAT_RATE).alias('vat_private_educ'),
        (pl.col('medicines_expenditures') * VAT_RATE).alias('vat_medicines'),
        (pl.col('private_health_expenditures') * VAT_RATE).alias('vat_private_health'),
        (pl.col('total_educ_health_expenditures') * VAT_RATE).alias('vat_total_educ_health'),
        ])
)

exempt_educ_health.write_csv('outputs/feb3/exempt_educ_health.csv')


# # Clean educ_health item codes - Add leading zero for 6-digit item codes; then get first 2 digits to classify major categories
# educ_health = educ_health.with_columns([
#     pl.col('item_code')
#       .cast(pl.Utf8)
#       .map_elements(clean_item_code, return_dtype=pl.Utf8)
#       .alias('new_item_code'),

#     pl.col('item_code')
#       .cast(pl.Utf8)
#       .map_elements(clean_item_code, return_dtype=pl.Utf8)
#       .map_elements(extract_major_code, return_dtype=pl.Utf8)
#       .alias('item_code_major')
# ])

print("Item codes cleaned and standardized")
print(educ_health[['item_code_major', 'new_item_code', 'item_code', 'description', 'vatable']].head())

# GROUP EDUCATION AND HEALTH INTO 3 GROUPS - EDUC SERVICES, HEALTH MEDICINES, HEALTH SERVICES
educ_health = educ_health.with_columns([
    pl.when(pl.col("item_code_major") == "10")
      .then(pl.lit("private_educ"))
      .when(pl.col("new_item_code").is_in(["0611122", "0611125", "0611126"]))
      .then(pl.lit("medicines"))
      .when(pl.col("new_item_code").is_in([
          "0621120", "0621929", "0623220",
          "0631020", "0632020", "0641010", "0641020"
      ]))
      .then(pl.lit("private_healthservices"))
      .otherwise(pl.lit("OTHER"))
      .alias("new_category")
])



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
        pl.sum('TOTEX').alias('total_FIES_expenditures'),
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
    .with_columns([
        (pl.col('RFACT') * pl.col('senior_count')).alias('weighted_number_seniors'),
        ((pl.col('total_vatable_expenditures') / pl.col('TOTEX') * 100).alias('ratio_vatable_to_total_expenditures'))
])
    .group_by('NPCINC')
    .agg([
        pl.sum('RFACT').alias('number_households_with_seniors'),
        pl.sum('senior_count').alias('number_seniors'),
        pl.sum('weighted_number_seniors').alias('weighted_number_seniors'),
        pl.sum('total_vatable_expenditures').alias('total_vatable_expenditures'),
        pl.sum('total_exempt_expenditures').alias('total_exempt_expenditures'),
        pl.sum('total_expenditures').alias('total_expenditures'),
        pl.sum('estimated_vat_perHH').alias('total_vat_paid'),
        pl.sum('estimated_vat_foregone').alias('total_vat_foregone'),
        # Mean VAT ETR by decile
        pl.mean('vat_paid_to_total_expenditures').alias('avg_vat_etr'),
        pl.mean('vat_foregone_to_total_expenditures').alias('avg_vat_foregone_etr'),
        pl.mean('ratio_vatable_to_total_expenditures').alias('avg_vat_to_total')
    ])
    .sort('NPCINC')
)

print(senior_summary)


# =============================================================================
# EXPORT RESULTS
# =============================================================================

# Export to separate CSVs
# tax_share.write_csv('outputs/feb3/tax_share.csv')
# tax_share_bydecile.write_csv('outputs/feb3/tax_share_bydecile.csv')
# budget_share.write_csv('outputs/feb3/budget_share.csv')
# main_summary.write_csv('outputs/feb3/main_summary.csv')
# exempt_by_category.write_csv('outputs/feb3/exempt_by_category.csv')
# vat_paid_and_foregone.write_csv('outputs/feb3/vat_paid_and_foregone.csv')
# senior_summary.write_csv('outputs/feb3/senior_summary.csv')

# Export processed household-level data
# fies_raw_vat_etr.write_parquet('clean_data/fies_raw_vat_etr.parquet')
# fies_raw_vat_etr.write_csv('clean_data/fies_raw_vat_etr.csv')   # Never mind. This creates 1 GB csv, not user friendly for processing
