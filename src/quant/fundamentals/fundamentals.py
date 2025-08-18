import os 
import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
#AV = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo'
FMP = f"https://financialmodelingprep.com/api/v3/search?query=AA&apikey=vQBysEwjaYxv3kgzUf3BfmPut94OEGHu?apikey={os.environ("FMP_API_KEY")}"
r = requests.get(url)
data = r.json()

print(data)

