import argparse
import pandas as pd

from config import CHART_DIR, LABELS_PATH
from market_data import download_ohlcv
from chart_generator import save_price_chart
from labeling import make_labels

def build_for_ticker(ticker, period, interval, window_size, step):
    df = download_ohlcv(ticker, period, interval)
    labels = make_labels(df)
    rows = []
    safe = ticker.replace("/", "_")

    for end in range(window_size, len(df) - 5, step):
        window = df.iloc[end-window_size:end]
        label = labels.iloc[end-1]
        if pd.isna(label):
            continue
        path = CHART_DIR / safe / f"{safe}_{end:06d}_{label}.png"
        save_price_chart(window, path, f"{ticker} | chart window")
        rows.append({
            "path": str(path.relative_to(CHART_DIR.parent)),
            "ticker": ticker,
            "label": label,
            "end_index": end
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    ap.add_argument("--period", default="5y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--window-size", type=int, default=60)
    ap.add_argument("--step", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for ticker in args.tickers:
        print("Generating", ticker)
        rows += build_for_ticker(ticker, args.period, args.interval,
                                 args.window_size, args.step)

    pd.DataFrame(rows).to_csv(LABELS_PATH, index=False)
    print("Saved", len(rows), "samples to", LABELS_PATH)

if __name__ == "__main__":
    main()
