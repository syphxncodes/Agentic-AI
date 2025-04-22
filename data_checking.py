import pandas as pd
import folium
from itertools import permutations
import numpy as np

# Load data
supply_data = pd.read_csv('synthetic_medical_supply_with_severity.csv')
medicines_data = pd.read_csv('medicines.csv')

# Redirect waste to DHA
waste_rows = supply_data[supply_data['weekly_wastage'] > 0].copy()
if not waste_rows.empty:
    waste_rows['hospital_name'] = 'DHA'
    waste_rows['quantity_supplied'] = waste_rows['weekly_wastage']
    supply_data.loc[waste_rows.index, 'quantity_supplied'] -= waste_rows['weekly_wastage']
    supply_data = pd.concat([supply_data, waste_rows], ignore_index=True)

# Hospital coordinates (replace with actual coordinates)
HOSPITAL_COORDS = {
    'Dubai Hospital': (25.2848034, 55.3215493),
    'Al Zahra Hospital Dubai': (25.1052654, 55.1804857),
    'Mediclinic City Hospital Dubai': (25.2303746, 55.3227651),
    'Thumbay Hospital Dubai': (25.0543598, 55.173236),
    'Saudi German Hospital Dubai': (25.0969334, 55.1838119),
    'NMC Royal Hospital Dubai': (25.0052466, 55.1556654),
    'American Hospital Dubai': (25.2355738, 55.3130763),
    'DHA': (25.2455633, 55.3097972)
}

def create_optimized_route_map(supplier, hospitals):
    # Create base map
    m = folium.Map(location=[25.2048, 55.2708], zoom_start=12)
    
    # Get valid coordinates
    valid_hospitals = [h for h in hospitals if h in HOSPITAL_COORDS]
    if not valid_hospitals:
        return None
    
    # Generate dummy optimized route (replace with actual routing logic)
    coords = [HOSPITAL_COORDS[h] for h in valid_hospitals]
    optimized_route = coords  # In real use, sort these coordinates
    
    # Add markers and path
    for idx, (hospital, coord) in enumerate(zip(valid_hospitals, optimized_route)):
        folium.Marker(
            coord,
            popup=f"{idx+1}. {hospital}",
            icon=folium.Icon(color='red' if hospital == 'DHA' else 'blue')
        ).add_to(m)
    
    folium.PolyLine(optimized_route, color='green', weight=2.5).add_to(m)
    
    return m

# Main processing
supplier_counts = supply_data.groupby('supplier_name')['hospital_name'].nunique()
multi_suppliers = supplier_counts[supplier_counts > 1].index.tolist()

medicine_quantity = medicines_data.set_index('name')['quantity'].to_dict()
supply_data['medicine_quantity'] = supply_data['supply_name'].map(medicine_quantity)
hospital_priority = supply_data.groupby('hospital_name')['medicine_quantity'].sum().sort_values().index.tolist()

for supplier in multi_suppliers:
    hospitals = supply_data[supply_data['supplier_name'] == supplier]['hospital_name'].unique()
    sorted_hospitals = sorted(hospitals, key=lambda x: hospital_priority.index(x))
    
    # Create and save map
    route_map = create_optimized_route_map(supplier, sorted_hospitals)
    if route_map:
        safe_name = supplier.replace(" ", "_").replace("/", "-")
        route_map.save(f"{safe_name}_route.html")

print("Maps generated successfully. Check the HTML files in your directory.")
