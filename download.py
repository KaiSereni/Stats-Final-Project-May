import pandas as pd
import json

# URLs to Wikipedia tables
median_age_url = "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_median_age"
gdp_url = "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_GDP"

# Read tables from Wikipedia
age_tables = pd.read_html(median_age_url)
gdp_tables = pd.read_html(gdp_url)

# Extract relevant tables
# For median age, the first table contains states and their median ages
age_df = age_tables[0]
# For GDP, the third table contains GDP per capita by state
gdp_df = gdp_tables[0]

# Print all column tuples to debug
print("All GDP column tuples:")
for col in gdp_df.columns:
    print(repr(col))

# Clean up age_df
age_df = age_df.rename(columns={age_df.columns[1]: "State", age_df.columns[2]: "Median Age"})
age_df = age_df[["State", "Median Age"]]
age_df = age_df[age_df["State"].isin([
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
    'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
    'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
    'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
    'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia'
])]
age_df = age_df.set_index("State")

print(gdp_df.columns[7])

# Clean up gdp_df
state_col = ('State or federal district', 'State or federal district')
gdp_per_capita_col = ('Nominal GDP per capita[1][2]', '2024')
gdp_df = gdp_df[[state_col, gdp_per_capita_col]]
gdp_df.columns = ['State', 'GDP per Capita']
gdp_df["GDP per Capita"] = gdp_df["GDP per Capita"].replace('[\$,]', '', regex=True).astype(float)
gdp_df = gdp_df[gdp_df["State"].isin(age_df.index)]
gdp_df = gdp_df.set_index("State")

# Merge dataframes
merged = age_df.join(gdp_df)

# Build the result dictionary
result = {}
for state, row in merged.iterrows():
    result[state] = {
        "avg_age": float(row["Median Age"]),
        "gdp_per_capita": float(row["GDP per Capita"])
    }

# Output as JSON
with open('out.json', 'w') as f:
    json.dump(result, f)