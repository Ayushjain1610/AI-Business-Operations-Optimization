import pandas as pd
from pulp import *

def optimize_production(data_path, forecast_path, output_path):
    df = pd.read_csv(data_path)
    forecast = pd.read_csv(forecast_path)

    forecast_demand = forecast.groupby('Product')['Forecasted_Demand'].sum().reset_index()
    forecast_demand.columns = ['Product','Forecast_Demand']

    cost_data = df.groupby('Product').agg({'Unit_Cost':'mean','Capacity':'mean'}).reset_index()
    data = forecast_demand.merge(cost_data, on='Product')

    model = LpProblem("Production_Optimization", LpMinimize)
    decision_vars = {}

    for _, row in data.iterrows():
        product = row['Product']
        decision_vars[product] = LpVariable(f"Q_{product}", lowBound=0)

    model += lpSum([row['Unit_Cost'] * decision_vars[row['Product']] for _, row in data.iterrows()])

    for _, row in data.iterrows():
        product = row['Product']
        model += decision_vars[product] >= row['Forecast_Demand']
        model += decision_vars[product] <= row['Capacity']

    model.solve()

    results = [[p, v.varValue] for p, v in decision_vars.items()]
    pd.DataFrame(results, columns=['Product','Optimal_Production']).to_csv(output_path, index=False)
