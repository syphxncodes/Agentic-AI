import csv
from io import StringIO
import pandas as pd
# Example data (replace with your actual data)
data=pd.read_csv("medicines.csv")
# Write to CSV string
csv_buffer = StringIO()
data.to_csv(csv_buffer, index=False)
csv_string = csv_buffer.getvalue()

print(csv_string)