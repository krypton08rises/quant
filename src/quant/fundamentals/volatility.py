import os 
import requests
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from eodhd import APIClient
 
def initialize_api(): 
    load_dotenv()

    api_key = os.getenv("EOD_API_KEY")
    eod_client = APIClient(api_key)
    return eod_client

def fetch_volatility(stock:str="NVDA", start_dt:str="2024-10-21", end_dt:str="2025-04-20"):


    eod_client = initialize_api()
    time_format = "%Y-%m-%d"
    # Fetching historical stock data
    stock_data = fetch_stock_data(
            eod_client, 
            stock, 
            datetime.strptime(start_dt, time_format), 
            datetime.strptime(end_dt, time_format))
    
    # Calculating daily returns
    stock_data['Daily_Return'] = stock_data['close'].pct_change()

    # Calculating volatility
    volatility = stock_data['Daily_Return'].std() * np.sqrt(252)

    # Plotting the volatility
    plt.figure(figsize=(10, 6))
    stock_data['Daily_Return'].plot()
    plt.title(f'Volatility of {stock_symbol}')
    plt.ylabel('Daily Returns')
    plt.xlabel('Date')
    plt.show()    


def fetch_stock_data(
        eod_client:eodhd.apiclient.APIClient,             
        symbol:str,  
        start_date:datetime.datetime, 
        end_date:datetime.datetime
    ): 
    data = eod_client.get_historical_data(
                symbol=symbol, 
                period="d", 
                from_date=start_date, 
                to_date=end_date,
                order='a'
            )
    return data

if __name__=="__main__":
    fetch_volatility()
