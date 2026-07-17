import pandas as pd
import numpy as np
def engineer_features (df):
    df=df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if "OpenInt" in df.columns:
        df = df.drop(columns=["OpenInt"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Quarter"] = df["Date"].dt.quarter
    df["Daily_Return"] = df["Close"].pct_change()
    df["Lag1"] = df["Close"].shift(1)
    df["Lag2"] = df["Close"].shift(2)
    df["Lag3"] = df["Close"].shift(3)
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["Rolling_STD"] = df["Close"].rolling(20).std()
    df["Rolling_Max"] = df["High"].rolling(20).max()
    df["Rolling_Min"] = df["Low"].rolling(20).min()
    df["Price_Range"] = df["High"] - df["Low"]
    df["OC_Diff"] = df["Close"] - df["Open"]
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["ROC_10"] = (
        (df["Close"] - df["Close"].shift(10))
        / df["Close"].shift(10)
    ) * 100
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * std
    df["BB_Lower"] = df["BB_Middle"] - 2 * std
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()
    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_5"] = df["Close"].pct_change(5)
    df["Return_20"] = df["Close"].pct_change(20)
    df["Volatility_20"] = (
    df["Daily_Return"]
      .rolling(20)
      .std()
)
    # df["Target"] = df["Close"].shift(-1)
    # df["Target"] = df["Close"].pct_change().shift(-1)
    df["Target"] = np.log(df["Close"]).diff().shift(-1)
    df = df.dropna().reset_index(drop=True)

    return df
FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Daily_Return",
    "Lag1",
    "Lag2",
    "Lag3",
    "SMA20",
    "EMA20",
    "Rolling_STD",
    "Rolling_Max",
    "Rolling_Min",
    "Price_Range",
    "OC_Diff",
    "Momentum_10",
    "ROC_10",
    "RSI",
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
    "MACD",
    "MACD_Signal",
    "ATR",
    "Return_1",
    "Return_5",
    "Return_20",
    "Volatility_20",
]