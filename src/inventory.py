import pandas as pd
import numpy as np

def calculate_inventory(data_path, output_path):
    df = pd.read_csv(data_path)

    demand_stats = df.groupby('Product')['Units_Sold'].agg(['mean','std']).reset_index()
    demand_stats.columns = ['Product','Avg_Demand','Demand_Std']

    lead_time = df.groupby('Product')['Lead_Time'].mean().reset_index()
    lead_time.columns = ['Product','Avg_Lead_Time']

    inventory = pd.merge(demand_stats, lead_time, on='Product')

    z = 1.65
    inventory['Safety_Stock'] = z * inventory['Demand_Std'] * np.sqrt(inventory['Avg_Lead_Time'])
    inventory['Reorder_Point'] = inventory['Avg_Demand'] * inventory['Avg_Lead_Time'] + inventory['Safety_Stock']

    inventory.to_csv(output_path, index=False)
