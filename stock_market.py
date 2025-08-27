import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import seaborn as sns

st.title("Stock Market Analysis and Prediction")

# st.header("1. Stock Data Retrieval")

# # Let's see example pf Widgets

# if st.checkbox("Show Instructions"):
#     st.write('This is a check box')
#     st.write('1. Enter a valid stock ticker symbol (e.g., AAPL for Apple Inc.).')
#     st.write('2. Select the start and end dates for the historical data.')  

start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('today'))

# symbol  = 'AAPL'  # Apple Inc. as default

# ticker_symbol = st.text_input("Enter Stock Ticker Symbol", ["AAPL","MSFT"])

ticker_symbol = st.selectbox("Select Stock Ticker Symbol", ["AAPL","MSFT","GOOGL","AMZN","TSLA","META"])

ticker_data = yf.Ticker(ticker_symbol)
ticker_df = ticker_data.history(start=start_date, end=end_date)


st.dataframe(ticker_df)

# Let's create some basic visualizations

st.write("## Closing Price Chart")
st.line_chart(ticker_df['Close']) # Inside a line chart, pass a series
st.write("## Volume Chart")
st.bar_chart(ticker_df['Volume'])

col1, col2 = st.columns(2)

with col1:
    st.write("## Moving Averages")
    ticker_df['MA20'] = ticker_df['Close'].rolling(window=20).mean()
    ticker_df['MA50'] = ticker_df['Close'].rolling(window=50).mean()
    plt.figure(figsize=(10,5))
    plt.plot(ticker_df['Close'], label='Closing Price')
    plt.plot(ticker_df['MA20'], label='20-Day MA')
    plt.plot(ticker_df['MA50'], label='50-Day MA')
    plt.title(f"{ticker_symbol} Closing Price and Moving Averages")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

with col2:
    st.write("## Daily Returns Distribution")
    ticker_df['Daily Return'] = ticker_df['Close'].pct_change()
    plt.figure(figsize=(10,5))
    sns.histplot(ticker_df['Daily Return'].dropna(), bins=50, kde=True)
    plt.title(f"{ticker_symbol} Daily Returns Distribution")
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    st.pyplot(plt)

st.cache_data.clear()
