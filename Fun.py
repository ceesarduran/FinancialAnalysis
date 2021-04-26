import yfinance as yf
import datetime
import pandas as pd
from pprint import *
import matplotlib.pyplot as plt


#CAPM
#1) Capital Assets Pricing Model (CAPM) is one of the most
#important models in Finance
#2)CAPM is a model that describes the relationship between the 
# expected return and risk of securities 
#3) CAPM indicates that the expected return on a security is equal to 
# the risk-free return plus a risk premium

#Risk Free Asset (rf)
#1) CAPM assumes that there exist a risk free asset with zero standard deviation
#2) A risk free asset could be a U.S government 10 year Treasury 

#Market Portfolio (rm)
#1) Market portfolio includes all securities in the market.
#2) The Market overall return is denoted as rm

#Beta (b)
#1)Beta is a meassure of the volatility or systematic risk of a security
#or portfolio compared to the entire market (S&P500)
# Tech Stocks generally have higher betas thann S&P500
#   b = 1; Strongly Correlated with the Market
#   b < 1; Less Volatile than the Market (Consumer (P&G))
#   b > 1; More Volatile than the Market 

#CAPM Formula
#ri = Expected return of a Security
# rm-rf = Risk premium (incentive for investing in a Risky Security)
# ri = rf + b(rm-rf)

stocks = ["GM", "RCL", "AAL" ,"MARA","GE","AMD","KO"]

#Function to Create Stock Data.
def BuildStocks(list):
    data = yf.download(stocks, period= "5y", interval = "1d", groupby = stocks)
    stocks_df = pd.DataFrame(data["Adj Close"]) 
    df = stocks_df.reset_index(level='Date', col_level=1, col_fill='')
    df.to_excel('StocksData.xlsx')
    return df



#Function to plot
def StocksPlot(data):
    data.plot(x = "Date" , y = stocks)
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Adjusted")
    plt.title("Stocks Price data")
    plt.show() 


#Function to normalizar
def normalizar(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]
    x.to_excel("Normalizado.xlsx")
    return x


#Function to determinate daily return
def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[1:]:
        for j in range (1, len(df)):
            df_daily_return[i][j] = ((df[i][j] - df[i][j-1])/df[i][j-1])*100
        df_daily_return[i][0]=0
    return df_daily_return

data = BuildStocks(stocks)
StocksPlot(data)
normalizado = normalizar(data)
StocksPlot(normalizado)
dailyreturn = daily_return(data)