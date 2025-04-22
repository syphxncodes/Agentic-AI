import pandas as pd
import numpy as np

old_suppliers = [
    'Rogers LLC', 'Cunningham, Yu and Smith', 'Sanchez, Smith and Graham',
    'Gray PLC', 'Johns Group', 'Rodriguez-Santana', 'Smith Ltd', 'Berry-Baker',
    'Garcia-Williams', 'Thomas, Tyler and Smith', 'Peterson-Williams',
    'Solis, Clark and Lynch', 'Adams Group', 'Mack LLC', 'Hunt, Cohen and Little'
]
uae_suppliers = [
    'Al Zahrawi Medical', 'Paramount Medical Equipment Trading LLC', 'Al Maqam Medical Supplies LLC',
    'Al Naghi Medical', 'Metromed', 'Kiwi Medical Supplies', 'Babirus LLC', 'AKI Pharma',
    'MPC Healthcare', 'City Pharmacy Co.', 'Pharma Solutions', 'Planet Distribution',
    'Gulf Drug', 'HealthPro Distributors UAE', 'Global Med UAE'
]
supplier_mapping = dict(zip(old_suppliers, uae_suppliers))
df = pd.read_csv("synthetic_medical_supply.csv")
if 'supplier_name' in df.columns:
    df['supplier_name'] = df['supplier_name'].replace(supplier_mapping)
df = df.drop("date_supplied", axis=1)

places = ["Dubai", "Sharjah", "Abu Dhabi", "Ras Al Khaimah", "Umm al quwain", "Fujairah", "Ajman"]
df['place'] = np.random.choice(places, size=len(df))
df['weekly_wastage'] = np.random.randint(10, 51, size=len(df))

# Add severity (scale 1-5, weighted distribution)
df['severity'] = np.random.choice([1, 2, 3, 4, 5], size=len(df), p=[0.1, 0.2, 0.4, 0.2, 0.1])

# Add number of people going to the hospital per week (range 100-1000)
df['people_per_week'] = np.random.randint(100, 1001, size=len(df))

# Add SHI per week (range 1-10)
df['shi_per_week'] = np.random.randint(1, 11, size=len(df))

# Update hospital names to UAE hospitals
dubai_hospitals = [
    'American Hospital Dubai',
    'Mediclinic City Hospital Dubai',
    'NMC Royal Hospital Dubai',
    'Saudi German Hospital Dubai',
    'Al Zahra Hospital Dubai',
    'Thumbay Hospital Dubai',
    'Emirates Hospital Dubai',
    'Dubai Hospital'
]

np.random.seed(42)  # For reproducibility
if 'hospital_name' in df.columns:
    df['hospital_name'] = np.random.choice(dubai_hospitals, size=len(df))
np.random.seed(42)  # For reproducibility
if 'hospital_name' in df.columns:
    df['hospital_name'] = np.random.choice(dubai_hospitals, size=len(df))
df=df.drop("shi_per_week",axis=1)
df=df.drop("place",axis=1)
df.to_csv("synthetic_medical_supply_with_severity.csv", index=False)
print(df.head())
print(df.info())
print(df["supply_name"].unique())