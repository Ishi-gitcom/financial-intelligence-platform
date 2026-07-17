from pathlib import Path
import joblib

from src.data_loader import load_stock_data
from src.feature_engineering import engineer_features
from src.config import DATA_DIR, MODEL_DIR
model = joblib.load(
    MODEL_DIR / "random_forest.pkl"
)

print("Model Loaded Successfully")
stock_data = load_stock_data(DATA_DIR)

apple = stock_data["aapl"]

apple = engineer_features(apple)
top_features = [

    "Return_5",
    "ROC_10",
    "Return_20",
    "Volatility_20",
    "Return_1",
    "RSI",
    "Daily_Return",
    "Volume",
    "OC_Diff",
    "MACD_Signal"

]
latest_data = apple[top_features].iloc[-1:]
prediction = model.predict(latest_data)

print("Predicted Tomorrow Return")

print(prediction[0])
pred = prediction[0]
confidence = min(abs(pred) * 5000, 100)
last_close = apple["Close"].iloc[-1]
expected_price = last_close * (1 + prediction)
if pred > 0.003:
    signal = "Strong Buy"

elif pred > 0:
    signal = "Buy"

elif pred < -0.003:
    signal = "Strong Sell"

else:
    signal = "Sell"
print()

print("Trading Signal :", signal)
print("="*60)

print("Stock : AAPL")

print("Today's Close :", apple["Close"].iloc[-1])
print("Predicted Return :", prediction[0])
print("Confidence :", round(confidence,2), "%")
print("Expected Price :", expected_price)
print("Signal :", signal)

print("="*60)