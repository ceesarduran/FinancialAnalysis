from pandas.core import groupby
import yfinance as yf
import datetime
import pandas as pd
from pandas import *
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from openpyxl import Workbook
from openpyxl import load_workbook
from pprint import *
import random 
import pandas as pd
import seaborn as sns
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go



stocks = ["SPY","TSLA","AAPL" ,"RCL", "AAL" ,"MARA","GE","AMD","KO", "AMZN"]
#tart = datetime.datetime(2016,1,1)
#end = datetime.datetime(2021,4,11)
data = yf.download(stocks, period= "5y", interval = "1d", groupby = stocks)
print(type(data))
stock_df = pd.DataFrame(data) 
#Borrado Columna Adj Close
stock_df = stock_df.drop([0])
stock_df = stock_df.drop('Close', 1)
stock_df = stock_df.drop('Low', 1)
stock_df = stock_df.drop('Open', 1)
stock_df = stock_df.drop('High', 1)
stock_df = stock_df.drop('Volume', 1)
#stock_df = stock_df.drop('Close', 1)
# saving the dataframe 
#stock_df.rename(columns={"" : "AAAAAAAAAAAPEPLWLD"})


stock_df.to_excel('BuildStocks.xlsx')
#stock_df.to_csv("BuidStock.csv")

stocks_df = stock_df.sort_values(by = ['Date'])
pprint("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
pprint(stocks_df)

data['Adj Close'].plot()
plt.grid()
plt.xlabel("Date")
plt.ylabel("Adjusted")
plt.title("Miscrosoft Price data")
#plt.style.use('dark_background')
print("---Plot graph finish---")
plt.ioff()
plt.show()


# def interactive_plot(df, title):
#     fig = px.line(title = title)
#     for i in df.columns[1:]:
#             fig.add_scatter(y = df[i], name = i)
#     fig.show()

# interactive_plot(stock_df,"Test")
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

def normalize(df):
    x = df.copy()
    for i in x.columns[0:]:
        x[i] = x[i]/x[i][0]
    return x

testa = normalize(data)
pprint(testa)
stockNor_df = normalize(stock_df)
print("4444444444444444444444499999999999999999")
pprint(stockNor_df)


testa['Adj Close'].plot()
plt.grid()
plt.xlabel("Date")
plt.ylabel("Adjusted")
plt.title("Miscrosoft Price data")
#plt.style.use('dark_background')
print("---Plot graph finish---")
plt.ioff()
plt.show()

testa.to_excel("Normalized.xlsx")

def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[0:]:
        for j in range (1, len(df)):
            df_daily_return[i][j] = ((df[i][j] - df[i][j-1])/df[i][j-1])*100
        df_daily_return[i][0]=0
    return df_daily_return

stocks_daily_return = daily_return(stock_df)
stocks_daily_return.to_excel("Daily_Return.xlsx")
pprint("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
pprint(stocks_daily_return)
pprint("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
pprint(type(stock_df))
pprint(type(data))
pprint(data.loc[:,"Adj Close"]["AAPL"])
pprint("ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
pprint(stocks_daily_return.loc[:,"Adj Close"]["TSLA"])

#def interactive_plot(df, title):
#    fig = px.line(title = title)
#    for i in df.columns[0:]:
#        fig.add_scatter(x = df['Date'], y = df[i], name = i)
#    fig.show()

#interactive_plot(stock_df,"Interactivo")
pprint("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
pprint("33333333333333333333333333333333333333333333333")
para = stocks_daily_return["Adj Close"]["AAPL"]
pprint(para)
pprint("33333333333333333333333333333333333333333333333")
julita = stocks_daily_return.loc[:,"Adj Close"]["AAPL"]
pprint("Valor de Julita: {}"  .format(julita))
julita2 = stocks_daily_return.loc[:,"Adj Close"]["TSLA"]
pprint("Valor de Julita: {}"  .format(julita2))
plt.scatter((stocks_daily_return.loc[:,"Adj Close"]["AAPL"]),(stocks_daily_return.loc[:,"Adj Close"]["TSLA"]))
#stocks_daily_return.plot(kind = "scatter" , x = julita , y = julita2)
pprint("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
#stocks_daily_return.loc[:,"Adj Close"].plot(kind = 'scatter', x = ["TSLA"], y = ["MARA"])
pprint("ggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg")
# stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = 'TSLA')
# stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = 'AMD')