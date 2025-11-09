import requests
import pandas as pd

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0",
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

session = requests.Session()
session.headers.update(headers)
session.get(
    "https://www.nseindia.com", 
    timeout=5, 
    headers={
        "User-Agent": "Mozilla/5.0"
})  # Initial request to set cookies

def nse_indices(index_name:str) -> pd.DataFrame:
    headers_home = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    session.get("https://www.nseindia.com", headers=headers_home, timeout=5)

    index_urls = {
        "nifty50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
        "banknifty": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20BANK",
        "nifty_midcap_250": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MIDCAP%20250",
    }
    if index_name not in index_urls:
        raise ValueError(f"Index '{index_name}' URL not defined")

    # First request to establish session
    session.get("https://www.nseindia.com/")
    
    # Actual request to API endpoint
    response = session.get(index_urls[index_name])
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data['data'])

    return df[['symbol', 'identifier', 'open', 'dayHigh', 'dayLow', 'lastPrice', 'previousClose']]

if __name__ == "__main__":
    # Example usage
    nifty50_df = fetch_index_constituents('nifty50')
    banknifty_df = fetch_index_constituents('banknifty')
    midcap250_df = fetch_index_constituents('nifty_midcap_250')

    # Save to CSVs
    nifty50_df.to_csv("nifty50.csv", index=False)
    banknifty_df.to_csv("banknifty.csv", index=False)
    midcap250_df.to_csv("midcap250.csv", index=False)

    print("✅ Data fetched and saved:")
    print(f" - nifty50.csv ({len(nifty50_df)} symbols)")
    print(f" - banknifty.csv ({len(banknifty_df)} symbols)")
    print(f" - midcap250.csv ({len(midcap250_df)} symbols)")