import streamlit as st
import joblib
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import plotly.express as px
from src.data_loader import load_single_stock
from src.feature_engineering import engineer_features
st.set_page_config(
    page_title="Financial Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Paths
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "Data" / "Raw" / "Stocks"

MODEL_PATH = BASE_DIR / "Models" / "random_forest.pkl"
# -------------------------------
# Get Available Stocks
# -------------------------------

stock_files = sorted(DATA_DIR.glob("*.txt"))

stock_list = [
    file.stem.replace(".us", "")
    for file in stock_files
]
# -------------------------------
# Load Model
# -------------------------------

model = joblib.load(MODEL_PATH)
st.write("✅ Model Loaded")
metrics = pd.read_csv(
    BASE_DIR / "results" / "model_comparison.csv"
)
SCALER_PATH = BASE_DIR / "Models" / "scaler.pkl"
FEATURE_PATH = BASE_DIR / "Models" / "top_features.pkl"

scaler = joblib.load(SCALER_PATH)
top_features = joblib.load(FEATURE_PATH)
# -------------------------------
# Load Data
# -------------------------------

# stock_data = load_single_stock(DATA_DIR)
# st.write("✅ Stock Data Loaded")
# st.write(f"Stocks Loaded: {len(stock_data)}")

# -------------------------------
# UI
# -------------------------------

st.title("📈 Financial Intelligence Platform")

st.markdown(
    """
### AI-Powered Stock Return Prediction

Predict next-day stock returns using Machine Learning,
technical indicators, and explainable AI.
"""
)

st.markdown("---")

# stock = st.selectbox(
#     "Choose Stock",
#     sorted(stock_data.keys())
# )
# ticker = st.text_input(
#     "Enter Stock Symbol",
#     value="AAPL"
# )
st.sidebar.title("📈 Financial Intelligence Platform")

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Settings")

# ticker = st.sidebar.text_input(
#     "Stock Symbol",
#     value="AAPL"
# )
ticker = st.sidebar.selectbox(
    "📈 Select Stock",
    stock_list,
    index=stock_list.index("AAPL") if "AAPL" in stock_list else 0
)
st.sidebar.markdown("---")
st.sidebar.success(f"Selected Stock: {ticker}")
st.info(f"Currently Analyzing: **{ticker.upper()}**")
st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Model")

st.sidebar.success("Random Forest")
st.sidebar.metric(
    "Stocks Supported",
    "7163"
)

st.sidebar.metric(
    "Features",
    len(top_features)
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Features")

st.sidebar.info("10 Technical Indicators")

st.sidebar.markdown("---")

st.sidebar.subheader("📅 Prediction")

st.sidebar.write("Next Trading Day")

st.sidebar.markdown("---")

st.sidebar.caption("Developed by Ishi Aggarwal")

try:
    # Load stock
    with st.spinner("Loading stock data..."):
        @st.cache_data
        def get_stock(ticker):
            return load_single_stock(DATA_DIR, ticker)
        df = get_stock(ticker)
        st.success(f"{ticker.upper()} Loaded")
        
    # Feature Engineering
        df = engineer_features(df)

    # Latest row
        latest = df.iloc[-1]

    # Prepare features
        X = latest[top_features].values.reshape(1, -1)

    # Prediction
        prediction = model.predict(X)[0]

    # Trading Signal
        if prediction > 0:
            signal = "🟢 BUY"
        elif prediction < 0:
            signal = "🔴 SELL"
        else:
            signal = "🟡 HOLD"

    # Closing prices
        last_close = latest["Close"]
        predicted_close = last_close * (1 + prediction)

    # Display
        st.markdown("---")

        col1, col2 = st.columns(2)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
        "Today's Close",
        f"${last_close:.2f}"
    )

        with col2:
            st.metric(
        "Predicted Close",
        f"${predicted_close:.2f}",
        delta=f"{prediction*100:.2f}%"
    )

        with col3:
            st.metric(
        "Signal",
        signal
    )
        st.markdown("---")

        st.subheader("📈 Last 60 Trading Days")

        chart_data = df.tail(60)

        fig = go.Figure()

# Close Price
        fig.add_trace(
        go.Scatter(
        x=chart_data["Date"],
        y=chart_data["Close"],
        mode="lines",
        name="Close",
        line=dict(width=3)
    )
)

# SMA20
        fig.add_trace(
        go.Scatter(
        x=chart_data["Date"],
        y=chart_data["SMA20"],
        mode="lines",
        name="SMA20"
    )
)

# EMA20
        fig.add_trace(
        go.Scatter(
        x=chart_data["Date"],
        y=chart_data["EMA20"],
        mode="lines",
        name="EMA20"
    )
)

        fig.update_layout(
    title=f"{ticker.upper()} Stock Price",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode="x unified",
    template="plotly_white",
    height=550
)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("📊 Technical Indicators")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
    "RSI",
    f"{latest['RSI']:.2f}",
    delta="Overbought" if latest["RSI"] > 70 else "Neutral"
)

            st.metric(
        "SMA20",
        f"${latest['SMA20']:.2f}"
    )
        with col2:
            st.metric(
        "EMA20",
        f"${latest['EMA20']:.2f}"
    )

            st.metric(
    "MACD",
    f"{latest['MACD']:.3f}",
    delta="Bullish" if latest["MACD"] > 0 else "Bearish"
)
        with col3:
            st.metric(
        "Volume",
        f"{int(latest['Volume']):,}"
    )

            st.metric(
        "Volatility",
        f"{latest['Volatility_20']:.4f}"
    )
        st.markdown("---")
        st.subheader("🧠 Model Explanation")

        explanations = []
        if latest["RSI"] > 70:
            explanations.append("🔴 RSI is above 70 (Overbought)")
        elif latest["RSI"] < 30:
            explanations.append("🟢 RSI is below 30 (Oversold)")
        else:
            explanations.append("🟡 RSI is in the neutral zone")
        if latest["MACD"] > 0:
            explanations.append("🟢 MACD is positive (Bullish Momentum)")
        else:
            explanations.append("🔴 MACD is negative (Bearish Momentum)")
        if latest["Close"] > latest["EMA20"]:
            explanations.append("🟢 Price is above EMA20")
        else:
            explanations.append("🔴 Price is below EMA20")
        if latest["Close"] > latest["SMA20"]:
            explanations.append("🟢 Price is above SMA20")
        else:
            explanations.append("🔴 Price is below SMA20")
        if latest["Return_5"] > 0:
            explanations.append("🟢 Recent 5-day return is positive")
        else:
            explanations.append("🔴 Recent 5-day return is negative")
        for item in explanations:
            st.write(item)
        st.markdown("---")

        # if signal == "BUY":
        #     st.success("✅ Overall Outlook: Bullish")
        # else:
        #     st.error("⚠️ Overall Outlook: Bearish")
        st.markdown("---")
        st.subheader("📌 Final Decision")

        if signal == "🟢 BUY":

            st.success("### 🟢 BUY Recommendation")

            st.write(
        """
The Random Forest model predicts a **positive return** for the next trading day.

Although some technical indicators may be mixed,
the machine learning model expects the stock to move upward.

**Recommendation:** Consider buying or holding the stock.
"""
    )

        elif signal == "🔴 SELL":

           st.error("### 🔴 SELL Recommendation")

           st.write(
        """
The Random Forest model predicts a **negative return** for the next trading day.

Even if several technical indicators remain bullish,
the model has identified patterns that historically
preceded short-term price declines.

**Recommendation:** Consider booking profits or waiting for confirmation.
"""
    )

        else:

          st.warning("### 🟡 HOLD Recommendation")

          st.write(
        """
The predicted return is close to zero.

The market appears uncertain.

**Recommendation:** Wait for stronger confirmation before taking action.
"""
    )
        st.markdown("---")
        st.subheader("🎯 Prediction Confidence")

        confidence = abs(prediction) * 100

        confidence = min(confidence * 20, 100)

        st.progress(confidence / 100)

        st.write(f"Confidence: **{confidence:.1f}%**")
        st.markdown("---")
        st.subheader("⚠️ Risk Level")

        vol = latest["Volatility_20"]

        if vol < 0.01:
            st.success("🟢 Low Risk")

        elif vol < 0.03:
            st.warning("🟡 Medium Risk")

        else:
            st.error("🔴 High Risk")
        st.markdown("---")
        st.subheader("⭐ Feature Importance")
        feature_names = joblib.load(
    BASE_DIR / "Models" / "top_features.pkl"
)

        importance = model.feature_importances_
        importance_df = pd.DataFrame({

    "Feature": feature_names,
    "Importance": importance

})
        importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)
        fig = px.bar(

    importance_df,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    title="Random Forest Feature Importance"

)

        fig.update_layout(

    yaxis=dict(categoryorder="total ascending")

)

        st.plotly_chart(
    fig,
    use_container_width=True
)
    
        st.divider()

        st.subheader("📊 Model Performance")

        rf_metrics = metrics[
    metrics["Model"] == "Random Forest"
    ].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
        "MAE",
        f"{rf_metrics['MAE']:.3f}"
    )

        with col2:
            st.metric(
        "RMSE",
        f"{rf_metrics['RMSE']:.3f}"
    )

        with col3:
            st.metric(
        "R²",
        f"{rf_metrics['R2']:.3f}"
    )
        st.info(
    """
These metrics are obtained using Time-Series Cross Validation,
which preserves chronological order and better reflects
real-world stock prediction performance.
"""
)
        report = pd.DataFrame({

    "Stock":[ticker],

    "Today's Close":[last_close],

    "Predicted Close":[predicted_close],

    "Predicted Return":[prediction],

    "Signal":[signal]

})
        csv = report.to_csv(index=False)

    st.download_button(

    "📥 Download Prediction Report",

    csv,

    "prediction_report.csv",

    "text/csv"

)
except FileNotFoundError:

    st.error("❌ Stock file not found.")

except KeyError:

    st.error("❌ Required feature missing.")

except Exception as e:

    st.error(
        f"⚠️ Unexpected Error:\n\n{e}"
    )
st.markdown("---")

st.caption(
    """
Financial Intelligence Platform • Version 1.0

Built using Python • Scikit-Learn • Streamlit • Plotly
"""
)
from datetime import datetime

st.caption(
    f"Prediction generated on: {datetime.now().strftime('%d %B %Y, %H:%M')}"
)
with st.expander("ℹ️ About this Model"):

    st.markdown("---")

    st.info(
f"""
### 📖 Why did the model predict this?

The prediction is based on multiple market features including:

• RSI (Momentum)

• MACD (Trend)

• Moving Averages

• Recent Returns

• Volatility

• Trading Volume

The Random Forest combines all these signals rather than relying on a single indicator.
"""
)