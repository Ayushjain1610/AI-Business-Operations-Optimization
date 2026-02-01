import pandas as pd

def calculate_profit(data_path, forecast_path, production_path, output_path):
    df = pd.read_csv(data_path)
    forecast = pd.read_csv(forecast_path)
    production = pd.read_csv(production_path)

    price_data = df.groupby('Product')['Price'].mean().reset_index()
    cost_data = df.groupby('Product')['Unit_Cost'].mean().reset_index()
    forecast_demand = forecast.groupby('Product')['Forecasted_Demand'].sum().reset_index()
    forecast_demand.columns = ['Product','Forecast_Demand']

    profit_df = production.merge(price_data, on='Product')
    profit_df = profit_df.merge(cost_data, on='Product')
    profit_df = profit_df.merge(forecast_demand, on='Product')

    profit_df['Revenue'] = profit_df['Optimal_Production'] * profit_df['Price']
    profit_df['Production_Cost'] = profit_df['Optimal_Production'] * profit_df['Unit_Cost']
    profit_df['Profit'] = profit_df['Revenue'] - profit_df['Production_Cost']

    profit_df.to_csv(output_path, index=False)
