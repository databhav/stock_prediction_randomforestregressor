import streamlit as st
import pandas as pd 
import numpy as np 
# USING YFINANCE TO SCRAPE DATA FROM THE INTERNET
import yfinance as yf
# IMPORTING THE MACHINE LEARNING ALGORITHM
from sklearn.ensemble import RandomForestRegressor
# IMPORTING R2_SCORE FOR ACCURACY
from sklearn.metrics import r2_score
from datetime import date
from plotly import graph_objs as go
from sklearn.model_selection import train_test_split


# SETTING THE TITLE OF THE PAGE
st.title("STOCK PREDICTION MODEL")
st.write('**DISCLAIMER:** Stocks are one of the most unpredictive investments, If you are considering investing in stocks, you should do your research and speak with a financial advisor and not rely on this model solely.')

# SELECTING THE STOCKS I WANT TO MONITOR 
stocks = ('HINDUNILVR.NS','RELIANCE.NS','TCS.NS','INFY','HDB')
selected_stocks = st.selectbox("Select stock name:", stocks)

# SETTING THE START DATE AND END DATE TO LOAD THE DATA BETWEEEN THE TIME FRAME
START = '2018-04-10'
TODAY = date.today().strftime('%Y-%m-%d')

# MAKING A FUNCTION TO LOAD THE DATA BASED ON SELECTED STOCK BETWEEN THE TIME FRAME
def load_data(ticker):
    data = yf.download(ticker, start=START,end=TODAY)
    data.reset_index(inplace=True)
    return data

# USING THE FUNCTION TO LOAD DATA FOR THE SELECTED STOCK
data_load_state = st.text('load data...')
data = load_data(selected_stocks)
data_load_state.text('loading data... done!')

# PRINTING THE TAIL OF THE LOADED DATA
st.subheader('Raw data')
st.write(data.tail(10))

# DEFINING A FUNCTION FOR PLOTTING OPENING AND CLOSING PRICE OVER THE YEARS
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close')) 
    fig.layout.update (xaxis_rangeslider_visible=True) 
    st.plotly_chart (fig)
st.write('##')
st.subheader('Historical data')
plot_raw_data()

# MAKING A DATAFRAME 'data2' WITH 'data' VALUES TO AVOID 'NaN' VALUES FOR AVOIDING ERROR
data2 = data
data2['Close_tom'] = data2['Close'].shift(-1)

# SPLITTING DATA INTO TRAINING AND TESTING SETS
train_data, test_data = train_test_split(data2, test_size=0.1, shuffle=False)
train_data2, test_data2 = train_test_split(data, test_size=0.1, shuffle=False)

# SELECTING FEATURES(X) AND TARGET VALUES(Y)
X_train = train_data[['Open','High','Close','Low','Volume']] 
y_train = train_data['Close_tom']
X_test = test_data[['Open','High','Close','Low','Volume']]
y_test = test_data2['Close']

# TRAINING THE MODEL
model = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=1)
model.fit(X_train,y_train)

# PREDICTING USING MODEL
pred = model.predict(X_test)
pred_df = pd.DataFrame(pred)

# PRINTING TOMORROWS PREDICTION
tomorrow = pred[-1]
st.write("Tomorrow's stock price will be: ", tomorrow)


data3 = pred_df = pd.DataFrame(pred)
data3 = pd.concat([test_data['Date'],test_data['Open'],test_data['Close']],axis=1)
data3 = data3.reset_index()
data3['ClosingPriceTomorrow'] = pred_df
data3['ClosingPriceTomorrow'] = data3['ClosingPriceTomorrow'].shift(1)


# PLOTTING ACTUAL VS PREDICTED VALUES
def plot_pred_data():
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data3['Date'], y=data3['ClosingPriceTomorrow'], name='prediction'))
    fig2.add_trace(go.Scatter(x=data3['Date'], y=data3['Close'], name='actual'))
    fig2.add_trace(go.Scatter(x=data3['Date'], y=data3['Open'], name='Opening price'))
    fig2.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig2)
st.write('##')
st.subheader('Predicted vs Actual data')
plot_pred_data()

# st.write(data3.tail(5)) WAS USED TO SEE IF ITS WORKING CORRECTLY AND FIGURING OUT THE ACCURACY OF THE PREDICTED AND ACTUAL VALUE OF THE SAME DAY

st.subheader('Accuracy score')
data3.dropna(inplace=True)      # REQUIRED SO WE DONT RUN INTO ERROR WHILE CALCULATING ACCURACY
# PRINTING THE ACCURACY SCORE TO KNOW HOW WELL IS OUR MODEL PERFORMING
st.write("Accuracy score of the predicted vs actual values: ",r2_score(data3['Close'],data3['ClosingPriceTomorrow']))

# PRINTING ACTUAL, PREDICTED AND OPENING PRICE OF THE TAIL OF TEST DATASET
data4 = pd.concat([test_data['Date'],test_data['Open'],test_data['Close']],axis=1)
data4 = data4.reset_index()
data4['ClosingPriceTomorrow'] = pred_df

st.subheader('Final data')
# PRINTING TO SEE TOMORROW'S STOCK PRICE PREDICTIONS
st.write(data4.tail(5))
