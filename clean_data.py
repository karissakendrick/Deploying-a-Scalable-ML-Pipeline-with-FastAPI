# Import necessary libraries.
import pandas as pd

# Load the census data.
census_df = pd.read_csv('data/census.csv')

# Remove spaces from the string columns.
for col in census_df.select_dtypes(include='object').columns:
    census_df[col] = census_df[col].str.strip()

# Save the cleaned data.
census_df.to_csv('data/census_cleaned.csv', index=False)

# Confirm data was saved.
print("Data cleaning complete. Saved to census_cleaned.csv")