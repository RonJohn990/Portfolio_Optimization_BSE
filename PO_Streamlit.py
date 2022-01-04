import streamlit as st
# import tweepy
# from textblob import TextBlob
# from wordcloud import WordCloud
import pandas as pd
import yfinance as yf
# import sqlalchemy
import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# import scipy
import altair as alt


style.use('ggplot')

# Ticker Lists and Sidebar---------------------------------------------------------------------------
tickerSensex = pd.read_csv('./data/BSE_Company_List.csv')
tickerSensex = tickerSensex.Symbol.to_list()
tickerSP = pd.read_csv('./data/SP_Company_List.csv')
tickerSP = tickerSP.Symbol.to_list()
tickerCrypto = pd.read_csv('./data/Crypto_List.csv')
tickerCrypto = tickerCrypto.Symbol.to_list()


st.sidebar.header('BSE Ticker')
ticker_selection = st.sidebar.multiselect("Entity Name", options = tickerSensex)

st.sidebar.header('S&P Ticker')
ticker_selection_sp = st.sidebar.multiselect("Entity Name", options = tickerSP)

st.sidebar.header('Crypto Ticker')
ticker_selection_crypto = st.sidebar.multiselect("Entity Name", options = tickerCrypto)

ticker_selection = ticker_selection + ticker_selection_sp + ticker_selection_crypto


# Creating a Date Range---------------------------------------------------------------------------
st.sidebar.header('Start Date')
start_date = dt.datetime(2007, 1, 10)
no_of_years = 10
dates = ['2007-01-01', '2016-01-01']
result = []
today = dt.date(2021, 6, 1)
current = dt.date(2007, 1, 1)    

while current <= today:
    result.append(current.isoformat())
    current += relativedelta(months=1)

result.reverse()
date = st.sidebar.selectbox('Year', result)


# Downloading Data from Yahooo Finance---------------------------------------------------------------------------
@st.cache
def market_data(tickers, dt):
    data = []
    #for ticker in tickers:
    tmp_data = yf.download(tickers, start = dt)
    tmp_data = tmp_data['Adj Close']
    tmp_data = tmp_data.dropna()
    #data.append(tmp_data)
    
    #data = pd.DataFrame(data, columns = tickers)
    return(tmp_data)



# Initiating the 3 tabs for Page 1---------------------------------------------------------------------------
tab1, tab2, tab3 = st.columns(3)

# Displaying Ticker Selection Data with an Expander----------------------------------------------------------
df_1 = market_data(ticker_selection, date)
stick_data_expander = st.expander('Display Data')
stick_data_expander.write(df_1)

# Selecting either Regualar percent change or Log change
ret1,ret2 = st.columns(2)
if ret1.checkbox('Percentage Change', value = True):
    df_returns_pct = df_1.pct_change()
    df_returns_pct = df_returns_pct.dropna()
    df_returns_pct = df_returns_pct.replace(np.Inf, -1)
    df_returns = df_returns_pct


if ret2.checkbox('Log Change'):
    def log_change(data):
        ret = np.diff(np.log(data))
        return ret
    df_returns = pd.DataFrame()
    df_returns = df_1.apply(log_change)
    df_returns.index = df_1.index[1:]



# Page 1 Tab 1 - Visualization ##############################################################
if tab1.button('Simple Visualization'):
    # Visualization
    ## Line Chart of Adj. Close Prices
    plt.plot(df_1)
    plt.legend(ticker_selection)
    st.subheader('Line Chart')
    st.line_chart(df_1)

    ## Histograms and PCT change line chart
    st.subheader('PCT change Line Chart')
    st.line_chart(df_returns)

    hist_data = (df_returns.to_numpy()).T
    group_labels = df_returns.columns

    ### Preprocessing and removing error values that might throwoff the histogram
    st.subheader("Histogram Plots")

    # Plotly Plot
    fig = go.Figure()
    for ticker in ticker_selection:
        fig.add_trace(go.Histogram(x = df_returns[ticker], name = ticker))
        fig.update_layout(barmode = 'overlay')
        fig.update_traces(opacity = 0.6)
    # fig = px.histogram(df_returns, nbins = 150, opacity = 0.6)
    st.plotly_chart(fig)

#############################################################################################



# Custom Portfolio Sidebar Input-------------------------------------------------------------
wts_custom = []
lock_toggle_ticker_list = []
custom_toggle = 0
if st.sidebar.checkbox('Custom Portfolio'):
    custom_toggle = 1
    for ticker in ticker_selection:
        tmp_no_input = st.sidebar.number_input(ticker, min_value = 0.0, max_value = 1.0)
        label_name = f'Lock {ticker}'
        if st.sidebar.checkbox(label = label_name):
            lock_toggle = 1
        else: 
            lock_toggle = 0
        wts_custom.append(tmp_no_input)
        lock_toggle_ticker_list.append(lock_toggle)

    st.write(lock_toggle_ticker_list)
    st.write(lock_toggle)


    RF = 0.02
    df_returns_mean = df_returns.mean() * 252
    annualized_returns_custom = np.sum(df_returns_mean * wts_custom)
    matrix_covariance_custom = df_returns.cov() * 252
    portfolio_variance_custom = np.dot(wts_custom, np.dot(matrix_covariance_custom, wts_custom))
    portfolio_std_custom = np.sqrt(portfolio_variance_custom)
    sharpe_custom = (annualized_returns_custom - RF) / portfolio_std_custom
    custom_portfolio_metrics = [annualized_returns_custom, portfolio_std_custom, sharpe_custom]

    # Output on main page Tab 2 -------------------------------------------------------------
    st.header('Custom Portfolio Metrics')  
    col1, col2, col3 = st.columns(3)
    col1.metric(label = 'Annualized Returns', value = np.round(custom_portfolio_metrics[0], 2))
    col2.metric(label = 'Annualized Risk',value = np.round(custom_portfolio_metrics[1], 2))
    col3.metric(label = 'Sharpe Ratio', value = np.round(custom_portfolio_metrics[2], 2))

# Page 1 Tab 2 -  ###########################################################################
if tab2.button('Portfolio Optimization'):
    lock_toggle = 0
    st.header('Efficient Frontier Analysis')

    # Defining function to plot the Efficient Frontier and Portfolio Metrics Dataframes-------
    def efficient_frontier(no_portfolios = 10, RF = 0.05, custom = 0, lock = 0):
        portfolio_returns = []
        portfolio_risk = []
        sharpe_ratio = []
        portfolio_weights = []
        df_returns_mean = df_returns.mean() * 252
        matrix_covariance = df_returns.cov() * 252
        
        i = 0
        for i in range(no_portfolios):
        # Generate Random Weights---------------------------------------------------------

            if lock == 1:
                weights = np.random.uniform(low = 0, high = 1 - wts_custom[2], size = 3)
                weights = np.round(weights / np.sum(weights), 3)
                # weights[2] = (wts_custom[2])
                portfolio_weights.append(weights)
            elif lock == 0:
                weights = np.random.random(len(ticker_selection))
                weights = np.round(weights / np.sum(weights), 3)
                portfolio_weights.append(weights)

            # Annualized Returns--------------------------------------------------------------
            annualized_returns = np.sum(df_returns_mean * weights) # Already annualized
            portfolio_returns.append(annualized_returns)

            # Covariance Matrix & Portfolio Risk Calc-----------------------------------------
            portfolio_variance = np.dot(weights.T, np.dot(matrix_covariance, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            portfolio_risk.append(portfolio_std)

            # Sharpe Ratio---------------------------------------------------------------------r
            sharpe = (annualized_returns - RF) / portfolio_std
            sharpe_ratio.append(sharpe)
    

        # Dataframe from Analysing all the simulated portfolios
        df_metrics = pd.DataFrame([np.array(portfolio_returns),
                                np.array(portfolio_risk),
                                np.array(sharpe_ratio),
                                str(np.array(portfolio_weights))], index = ['Returns', 'Risk', 'Sharpe_ratio', 'Weights'])

        df_metrics = df_metrics.T
        
        # Finfing the min risk portfolio
        min_risk = df_metrics.iloc[df_metrics['Risk'].astype(float).idxmin()]
        wt_min  = portfolio_weights[df_metrics['Risk'].astype(float).idxmin()]
        # Finding the max return portfolio    
        max_return = df_metrics.iloc[df_metrics['Returns'].astype(float).idxmax()]
        wt_max  = portfolio_weights[df_metrics['Returns'].astype(float).idxmin()]
        # Finding the portfolio with max Sharpe Ratio    
        max_sharpe = df_metrics.iloc[df_metrics['Sharpe_ratio'].astype(float).idxmax()]
        wt_sharpe  = portfolio_weights[df_metrics['Sharpe_ratio'].astype(float).idxmax()]
        # Isolating the weights from the above mentioned 3 portfolios
        weights = pd.DataFrame([wt_min, wt_max, wt_sharpe], index = ['Min Risk', 'Max Return', 'Max Sharpe Ratio'], columns = ticker_selection)


       # Plotting the Efficient Frontier -------------------------------------------------------------------------
        df_metrics['Sharpe Ratio'] = df_metrics['Sharpe_ratio'].to_numpy(dtype = float)
    
        fig = go.Figure()
        c = fig.add_trace(go.Scatter(x = df_metrics['Risk'], y = df_metrics['Returns'], mode = 'markers'))
        
        if custom == 1:
            d = fig.add_trace(go.Scatter(x = (custom_portfolio_metrics[1], np.nan) , y = (custom_portfolio_metrics[0], np.nan), mode = 'markers'))
            #st.plotly_chart(c)
            st.plotly_chart(d)

        elif custom == 0:
            st.plotly_chart(c)
     

        return min_risk, max_return, max_sharpe, portfolio_weights, weights

    # Executing efficient_frontier funciton---------------------------------------------------------------------------
    a_exec = efficient_frontier(250, 0.02, custom_toggle, lock_toggle) 

    min_risk = pd.DataFrame({'Min Risk' : a_exec[0]})
    max_return = pd.DataFrame({'Max Return' : a_exec[1]})
    max_sharpe = pd.DataFrame({'Max Sharpe Ratio' : a_exec[2]})

    final_portfolios_df = pd.concat([min_risk, max_return, max_sharpe], axis = 1)
    weights_df = pd.DataFrame(a_exec[3], columns = ticker_selection)
    final_portfolios_df = final_portfolios_df.drop('Weights', axis = 0)

    st.subheader('Weights of the required Portfolios')
    st.dataframe(a_exec[4])
    
    st.subheader("Portfolio Metrics")
    st.write(final_portfolios_df)
######################################################################################################################




# Page  1 Tab 3 -  ###################################################################################################
if tab3.button('Twitter Analysis'):
    st.header('Qualitative Analysis')
    st.selectbox('Ticker', ticker_selection)
    
#######################################################################################################################