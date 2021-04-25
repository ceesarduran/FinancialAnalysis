import yfinance as yf
import datetime
import pandas as pd
from pprint import *
import matplotlib.pyplot as plt

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