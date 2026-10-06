def make_labels(df, horizon=5, buy_threshold=0.02, sell_threshold=-0.02):
    future = df["Close"].shift(-horizon) / df["Close"] - 1.0
    labels = future.map(
        lambda x: "BUY" if x >= buy_threshold
        else "SELL" if x <= sell_threshold
        else "HOLD"
    )
    return labels
