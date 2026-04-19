"""
Fetch live NSE index constituents and write them to src/quant/data/indices/.

Each index is saved as ``<slug>.csv`` with a ``Symbol`` column (matches the
schema consumed by ``kite_data.backfill_index``).
"""

from pathlib import Path

import pandas as pd
import requests

INDICES_DIR = Path(__file__).resolve().parent / "indices"

INDEX_URLS = {
    # Broad market
    "nifty_50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
    "nifty_next_50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20NEXT%2050",
    "nifty_100": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100",
    "nifty_200": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20200",
    "nifty_500": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500",
    "midcap_150": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MIDCAP%20150",
    # NIFTY MIDCAP 250 is not exposed by the stockIndices API (returns {});
    # use nifty_500 for broader mid/small coverage.
    "smallcap_100": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20SMALLCAP%20100",
    "smallcap_250": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20SMALLCAP%20250",
    # Sectoral / thematic
    "banknifty": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20BANK",
    "nifty_private_bank": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20PRIVATE%20BANK",
    "nifty_psu_bank": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20PSU%20BANK",
    "nifty_finserv": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20FINANCIAL%20SERVICES",
    "nifty_it": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20IT",
    "nifty_auto": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20AUTO",
    "nifty_pharma": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20PHARMA",
    "nifty_fmcg": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20FMCG",
    "nifty_metal": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20METAL",
    "nifty_energy": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20ENERGY",
    "nifty_realty": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20REALTY",
    "nifty_consumption": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20INDIA%20CONSUMPTION",
    "nifty_infra": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20INFRASTRUCTURE",
    "nifty_media": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MEDIA",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}

# NSE's root returns 403; this page seeds the anti-bot cookies (nsit, _abck, bm_sz)
# that the stockIndices API requires.
_COOKIE_SEED_URL = "https://www.nseindia.com/market-data/live-equity-market"


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_HEADERS)
    session.get(_COOKIE_SEED_URL, timeout=10)
    return session


def nse_indices(index_name: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch constituents for a single NSE index. Strips the index-header row NSE prepends."""
    if index_name not in INDEX_URLS:
        raise ValueError(f"Index '{index_name}' not defined. Available: {sorted(INDEX_URLS)}")
    session = session or _make_session()
    response = session.get(INDEX_URLS[index_name], timeout=15)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["data"])
    # NSE prepends a summary row for the index itself (priority=1); constituents are priority=0.
    if "priority" in df.columns:
        df = df[df["priority"] == 0].reset_index(drop=True)
    return df


def save_constituents(index_name: str, session: requests.Session | None = None) -> Path:
    """Fetch and write one index's constituents to ``INDICES_DIR/<index_name>.csv``."""
    df = nse_indices(index_name, session=session)
    df = df.rename(columns={"symbol": "Symbol"})
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    out = INDICES_DIR / f"{index_name}.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    session = _make_session()
    for name in INDEX_URLS:
        try:
            out = save_constituents(name, session=session)
            df = pd.read_csv(out)
            print(f" - {out.name} ({len(df)} symbols)")
        except Exception as exc:
            print(f" ! {name}: {exc}")
