import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from yahoo_fin import news

# Streamlit UI Setup
st.title(" Stock Market Prediction with XGBoost & Latest News")

# Predefined list of popular stock tickers
popular_stocks = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "NFLX"]

# Sidebar for stock selection with search bar
st.sidebar.header("Stock Selection")
selected_stock = st.sidebar.selectbox("Select a stock or type a ticker:", popular_stocks, index=0)
custom_stock = st.sidebar.text_input("Or enter a custom stock ticker:", "")
if custom_stock:
    selected_stock = custom_stock.upper()

days_to_predict = st.sidebar.slider("Days to Predict", 1, 30, 7)

# Fetch the latest stock data
stock_data = yf.download(selected_stock, period="3y", interval="1d", progress=False)
stock_data.index = pd.to_datetime(stock_data.index)

if not stock_data.empty:
    latest_date = stock_data.index[-1]
    st.write(f" Latest Stock Data Date: {latest_date.strftime('%Y-%m-%d')}")
else:
    st.error("\u26a0\ufe0f No recent stock data found!")

# Plot Stock Price
fig, ax = plt.subplots()
ax.plot(stock_data["Close"], label="Closing Price", color='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title(f"{selected_stock} Stock Price Over Time")
ax.legend()
st.pyplot(fig)

# Prepare Data for XGBoost
stock_data['Prediction'] = stock_data['Close'].shift(-days_to_predict)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(stock_data[['Close']])
X = data_scaled[:-days_to_predict]
y = stock_data['Prediction'].dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# Predict Future Prices
future_prices = model.predict(X_test)
mae = mean_absolute_error(y_test, future_prices)
st.write(f" Mean Absolute Error: {mae:.2f} USD")

# Display Predictions
prediction_start_date = stock_data.index[-1] + pd.Timedelta(days=1)
pred_dates = pd.date_range(start=prediction_start_date, periods=len(future_prices), freq='B')
pred_df = pd.DataFrame({"Date": pred_dates, "Predicted Price": future_prices})
st.subheader(" Stock Price Predictions")
st.write(pred_df)

# Plot Predictions
fig2, ax2 = plt.subplots()
ax2.plot(stock_data.index, stock_data["Close"], label="Historical Price", color='blue')
ax2.plot(pred_dates, future_prices, label="Predicted Price", color='red')
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.set_title(f"{selected_stock} Price Prediction")
ax2.legend()
st.pyplot(fig2)

# Fetch and Display Latest News from Yahoo Finance
st.subheader(f" Latest News for {selected_stock}")
try:
    stock_news = news.get_yf_rss(selected_stock)
    if stock_news:
        for article in stock_news[:5]:
            title = article.get('title', 'No Title')
            link = article.get('link', '#')
            source = article.get('publisher', 'Unknown Source')
            pub_date = article.get('pubDate', 'Unknown Date')
            description = article.get('summary', 'No description available.')
            
            st.markdown(f"""
                ###  [{title}]({link})  
                ** Source:** {source}  
                ** Published on:** {pub_date}  
                ** Summary:** {description}  
                ---
            """, unsafe_allow_html=True)
    else:
        st.warning(" No recent news found for this stock.")
except Exception as e:
    st.error("Error fetching news.")

st.success("Prediction & News Display Completed! ")
