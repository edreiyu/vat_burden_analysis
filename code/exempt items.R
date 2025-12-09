

library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)

# ---- 1. Recreate the dataset from screenshot ----

df <- tribble(
  ~NPCINC, ~food_exempt, ~housing_utils_exempt, ~furnishings_exempt,
  ~health_exempt, ~transport_exempt, ~recreation_exempt,
  ~education_exempt, ~insurance_exempt,
  1, 72, 17, 0, 1, 5, 0, 4, 1,
  2, 67, 21, 0, 1, 5, 0, 5, 1,
  3, 63, 23, 0, 1, 6, 0, 5, 2,
  4, 59, 26, 0, 2, 6, 0, 5, 3,
  5, 56, 27, 0, 2, 6, 0, 5, 4,
  6, 52, 29, 0, 2, 7, 0, 5, 4,
  7, 49, 30, 1, 3, 7, 0, 6, 5,
  8, 45, 31, 1, 3, 7, 0, 6, 6,
  9, 39, 34, 1, 4, 6, 0, 7, 7,
 10, 28, 39, 5, 6, 4, 0, 9, 9
)

# ---- 2. Convert to long format for ggplot ----

df_long <- df %>%
  pivot_longer(
    cols = -NPCINC,
    names_to = "category",
    values_to = "share"
  )

# Make cleaner labels
category_labels <- c(
  food_exempt = "Food",
  housing_utils_exempt = "Housing & Utilities",
  furnishings_exempt = "Furnishings",
  health_exempt = "Health",
  transport_exempt = "Transport",
  recreation_exempt = "Recreation",
  education_exempt = "Education",
  insurance_exempt = "Insurance & Financial"
)

df_long$category <- factor(df_long$category, levels = names(category_labels), labels = category_labels)

# ---- 3. Create side-by-side small multiples ----

ggplot(df_long, aes(x = NPCINC, y = share)) +
  geom_col(fill = "#orange") +  # choose a single neutral color
  facet_wrap(~ category, scales = "free_y", ncol = 4) +
  labs(
    title = "Composition of Household Spending on VAT-Exempt Goods/Services",
    subtitle = "Shares by Per Capita Income Decile",
    x = "Income Decile (NPCINC)",
    y = "Share of VAT-Exempt Spending (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )



library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)

# --- Recreate data from screenshot ---
df <- tribble(
  ~NPCINC, ~food_exempt, ~housing_utils_exempt, ~furnishings_exempt,
  ~health_exempt, ~transport_exempt, ~recreation_exempt,
  ~education_exempt, ~insurance_exempt,
  1, 72, 17, 0, 1, 5, 0, 4, 1,
  2, 67, 21, 0, 1, 5, 0, 5, 1,
  3, 63, 23, 0, 1, 6, 0, 5, 2,
  4, 59, 26, 0, 2, 6, 0, 5, 3,
  5, 56, 27, 0, 2, 6, 0, 5, 4,
  6, 52, 29, 0, 2, 7, 0, 5, 4,
  7, 49, 30, 1, 3, 7, 0, 6, 5,
  8, 45, 31, 1, 3, 7, 0, 6, 6,
  9, 39, 34, 1, 4, 6, 0, 7, 7,
 10, 28, 39, 5, 6, 4, 0, 9, 9
)

# --- Convert to long format ---
df_long <- df %>%
  pivot_longer(
    cols = -NPCINC,
    names_to = "category",
    values_to = "share"
  ) %>%
  mutate(
    category = factor(category,
                      levels = c("food_exempt", "housing_utils_exempt", "furnishings_exempt",
                                 "health_exempt", "transport_exempt", "recreation_exempt",
                                 "education_exempt", "insurance_exempt"),
                      labels = c("Food", "Housing & Utilities", "Furnishings",
                                 "Health", "Transport", "Recreation",
                                 "Education", "Insurance & Financial")
    )
  )

# --- Small multiples plot ---
ggplot(df_long, aes(x = NPCINC, y = share)) +
  geom_col(fill = "#4C9F70") +
  geom_text(aes(label = paste0(share, "%")),
            vjust = -0.4, size = 3) +
  facet_wrap(~ category, scales = "fixed", ncol = 4) +  # uniform y-axis
  scale_x_continuous(breaks = 1:10, labels = 1:10) +
  labs(
    title = "VAT-Exempt Expenditure Composition by Income Decile",
    subtitle = "Side-by-side small multiples showing category shares (%)",
    x = "Per Capita Income Decile",
    y = "Share of VAT-Exempt Spending (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 9)
  )
