import yfinance as yf
import datetime
import pandas as pd
from pprint import *
import matplotlib.pyplot as plt

stocks = ["GM", "RCL", "AAL" ,"MARA","GE","AMD","KO"]

#Function to Create Stock Data.
def BuildStocks(list):
    start = datetime.datetime(2016,1,1)
    end = datetime.datetime(2021,4,11)
    data = yf.download(stocks, start=start, end=end)
    df = pd.DataFrame(data) 
    #Borrado Columna Adj Close
    df = df.drop('Close', 1)
    df = df.drop('Low', 1)
    df = df.drop('Open', 1)
    df = df.drop('High', 1)
    # saving the dataframe 
    df.to_excel('StocksData.xlsx')
    return data

#data = BuildStocks(stocks)

#Function to plot
def StocksPlot(data):
    data['Adj Close'].plot()
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Adjusted")
    plt.title("Miscrosoft Price data")
    plt.style.use('dark_background')
    plt.show() 

#Function to normalizar
def normalizar(df):
    x = df.copy()
    for i in x.columns[0:]:
        x[i] = x[i]/x[i][0]
    x.to_excel("Normalizado.xlsx")
    return x


#Function to determinate daily return
def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[0:]:
        for j in range (1, len(df)):
            df_daily_return[i][j] = ((df[i][j] - df[i][j-1])/df[i][j-1])*100
        df_daily_return[i][0]=0
    return df_daily_return

