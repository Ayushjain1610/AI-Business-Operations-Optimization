import pandas as pd
from prophet import Prophet

def generate_forecast(data_path, output_path):
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])

    daily_sales = df.groupby(['Date','Product'])['Units_Sold'].sum().reset_index()
    all_forecasts = []

    for product in daily_sales['Product'].unique():
        data = daily_sales[daily_sales['Product']==product][['Date','Units_Sold']].copy()
        data.columns = ['ds','y']

        model = Prophet()
        model.fit(data)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        temp = forecast[['ds','yhat']].copy()
        temp.columns = ['Date','Forecasted_Demand']
        temp['Product'] = product
        all_forecasts.append(temp)

    pd.concat(all_forecasts).to_csv(output_path, index=False)
