import pandas as pd
import numpy as np
df=pd.read_csv("synthetic_medical_supply_with_severity.csv")
json_ready = (
    df.groupby(['hospital', 'supply_name'])['quantity_supplied']
    .mean()
    .round(2)
    .reset_index()
    .rename(columns={'quantity_supplied': 'avg_quantity'})
    .to_dict(orient='records')
)
