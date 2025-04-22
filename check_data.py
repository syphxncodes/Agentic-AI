import pandas as pd

# Load the datasets
df = pd.read_csv("synthetic_medical_supply_with_severity.csv")
df1 = pd.read_csv("medicines.csv")    # Extra medicines
df2 = pd.read_csv("equipment.csv")    # Extra equipment

# Combine medicines and equipment into one DataFrame
df_extra = pd.concat([df1, df2], ignore_index=True)

# Ensure column names match
df_extra = df_extra.rename(columns={'name': 'supply_name', 'quantity': 'extra_quantity'})

# Merge the extra supplies onto the original df by supply_name and hospital_name
df = df.merge(
    df_extra,
    on='supply_name',
    how='left'
)

# Replace NaNs in extra_quantity with 0 (i.e., no extra supply if not found)
df['extra_quantity'] = df['extra_quantity'].fillna(0)

# Update quantity_supplied with the extra quantity
df['quantity_supplied'] += df['extra_quantity']

# Drop the temporary extra_quantity column
df.drop(columns=['extra_quantity'], inplace=True)

# Save updated DataFrame if needed
df.to_csv("updated_medical_supply.csv", index=False)
print("✅ Quantities successfully updated and saved to 'updated_medical_supply.csv'")
