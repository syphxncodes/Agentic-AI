import pandas as pd
import random

# Define lists of suppliers, hospitals, and base supply names
suppliers = ["MedSupply Inc.", "HealthPro Distributors", "Global Med", "CarePlus", "MediWorld"]
hospitals = ["City Medical Center", "Riverside Hospital", "Green Valley Hospital", "Sunrise Clinic", "Lakeside Medical"]
supply_base_names = ["Surgical Masks", "IV Drip Sets", "Syringes", "Gloves", "Bandages", "Thermometers", "Stethoscopes", "Wheelchairs", "Oxygen Tanks", "Defibrillators"]

# Generate 5,000 unique supply names
supply_names = [f"{name} Model {i}" for i in range(1, 501) for name in supply_base_names]  # 10 base names * 500 = 5000

# Randomly assign suppliers, hospitals, and quantities
random.seed(42)  # For reproducibility

data = []
for i in range(5000):
    data.append({
        "Supplier_to_hospital": random.choice(suppliers),
        "Hospital Name": random.choice(hospitals),
        "Supply Name": supply_names[i],
        "Quantity": random.randint(50, 5000)
    })

# Create DataFrame and save to CSV
large_df = pd.DataFrame(data)
large_df.to_csv("supplier_to_hospital_supplies_5000.csv", index=False)

# Display the first few rows
print(large_df.head())
