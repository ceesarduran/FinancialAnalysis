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
import pandas as pd
import seaborn as sns
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go





stocks = ["AAPL" ,"RCL", "AAL" ,"MARA","GE","KO", "AMZN","AMD","TSLA","MELI","^GSPC"]
#stocks = []

#def Stock_Analysis():
 #   print("Introduzca la cantidad de acciones a Analizar:")
 #   n = input()
 #   for i in range(n):
 #       print(i") ")
 #       s = input()
 #       stocks.append(s)
 #   print(stocks)
 #   return stocks

#stocks.append("^GSPC")
#tart = datetime.datetime(2016,1,1)
#end = datetime.datetime(2021,4,11)
data = yf.download(stocks, period= "5y", interval = "1d", groupby = stocks)
print(type(data))
stocks_df = pd.DataFrame(data["Adj Close"]) 
df2 = stocks_df.reset_index(level='Date', col_level=1, col_fill='')


#ls_key = "Adj Close"
#cleanData = data.loc[ls_key]

#stock_df = pd.DataFrame(data)
#aaa_df = stock_df.DataFrame(cleanData)  

#aaa_df = pd.DataFrame(cleanData) 

#aaa = aaa_df.reset_index(level='Date', col_level=1, col_fill='')
#aaa_df.to_excel('PLOMO.xlsx')
#leanData = stock_df(ls_key)


#Borrado Columna Adj Close
#stock_df = stock_df.drop([0])
#stock_df = stock_df.drop('Close', 1)
#stock_df = stock_df.drop('Low', 1)
#stock_df = stock_df.drop('Open', 1)
#stock_df = stock_df.drop('High', 1)
#stock_df = stock_df.drop('Volume', 1)
#stock_df = stock_df.drop('Close', 1)
# saving the dataframe 
#stock_df.rename(columns={"" : "AAAAAAAAAAAPEPLWLD"})


#stock_df.to_excel('BuildStocks.xlsx')
#stock_df.to_csv("BuiStock.csv")

#stocks_df = stock_df.sort_values(by = ['Date'])

#df2 = stocks_df.reset_index(level='Date', col_level=1, col_fill='')
#df3 = df2.drop(index='Adj Close', level=0)
#stocks_df = stocks_df.reset_index()

#nan_value = float("NaN")
#stock_df.replace("", nan_value, inplace=True)

#df2 = stocks_df.tail(-1)
#stocks_df = stocks_df.reset_index()
#stocks_df.to_excel('CharGay.xlsx')

df2.to_excel("BuildStocks.xlsx")

#df3.to_excel("Julita.xlsx")


pprint("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
pprint(stocks_df)

df2.plot(x = "Date" , y = stocks)
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
    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]
    return x

testa = normalize(df2)
pprint(testa)
#stockNor_df = normalize(stocks_df)
print("4444444444444444444444499999999999999999")
#pprint(stockNor_df)


testa.plot(x = "Date" , y = stocks)
plt.grid()
plt.xlabel("Date")
plt.ylabel("Adjusted")
plt.title("Stocks Price data")
#plt.style.use('dark_background')
print("---Plot graph finish---")
plt.ioff()
plt.show()

testa.to_excel("Normalized.xlsx")

def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[1:]:
        for j in range (1, len(df)):
            df_daily_return[i][j] = ((df[i][j] - df[i][j-1])/df[i][j-1])*100
        df_daily_return[i][0]=0
    return df_daily_return

stocks_daily_return = daily_return(df2)
stocks_daily_return.to_excel("Daily_Return.xlsx")
pprint("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
pprint(stocks_daily_return)
pprint("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
pprint(type(stocks_df))
pprint(type(data))
#pprint(data["AAPL"])
pprint("ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
#pprint(stocks_daily_return.loc[:,"Adj Close"]["TSLA"])

#def interactive_plot(df, title):
#    fig = px.line(title = title)
#    for i in df.columns[0:]:
#        fig.add_scatter(x = df['Date'], y = df[i], name = i)
#    fig.show()

#interactive_plot(stock_df,"Interactivo")
pprint("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
pprint("33333333333333333333333333333333333333333333333")
para = stocks_daily_return["AAPL"]
pprint(para)
pprint("33333333333333333333333333333333333333333333333")
#julita = stocks_daily_return.loc[:,"Adj Close"]["AAPL"]
#pprint("Valor de Julita: {}"  .format(julita))
#julita2 = stocks_daily_return.loc[:,"Adj Close"]["TSLA"]
#pprint("Valor de Julita: {}"  .format(julita2))
#plt.scatter((stocks_daily_return.loc[:,"Adj Close"]["AAPL"]),(stocks_daily_return.loc[:,"Adj Close"]["TSLA"]))
#stocks_daily_return.plot(kind = "scatter" , x = julita , y = julita2)
#pprint("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
#stocks_daily_return.loc[:,"Adj Close"].plot(kind = 'scatter', x = ["TSLA"], y = ["MARA"])
pprint("ggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg")
# stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = 'TSLA')
# stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = 'AMD')

charlie = daily_return(df2)
pprint(charlie)
pprint(charlie["AAPL"])
pprint("sssssssssssssssssssssssssssssssssssssssss")
#pprint(charlie["RCL"])
#plt.scatter( x = charlie["Adj Close"]["RCL"], y = charlie["Adj Close"]["AAL"]);
charlie.plot(kind = 'scatter', x = '^GSPC', y = 'AAPL')
plt.grid()
plt.show()

beta, alpha = np.polyfit(charlie['^GSPC'], stocks_daily_return['AAPL'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('AAPL', beta, alpha))

#charlie.plot(kind = 'scatter', x = 'SPY', y = 'AAPL')

# Straight line equation with alpha and beta parameters 
# Straight line equation is y = beta * rm + alpha
#charlie.plot(charlie['SPY'], beta * charlie['SPY'] + alpha, '-', color = 'r')
charlie.plot(kind = 'scatter', x = '^GSPC', y = 'AAPL')
plt.plot(charlie['^GSPC'], beta * charlie['^GSPC'] + alpha, '-', color = 'r')

plt.grid()
plt.show()


print("Promedio de SP500 {}" .format(charlie["^GSPC"].mean()))

rm = charlie["^GSPC"].mean() * 252


print("Promedio de Retorno anual: {}" .format(rm))

rf = 0

ER_AAPL = rf + (beta*(rm-rf))

print("Return of Apple: {}" .format(ER_AAPL ))
#plt.scatter

def interactive_plot(df, title):
    fig = px.line(title = title)
    for i in df.columns[1:]:
        fig.add_scatter(x = df['Date'], y = df[i], name = i)
    fig.show()
    
interactive_plot(df2, "Interactive")
interactive_plot(testa, "Interactive Normalized")

beta = {}
alpha = {}

# Loop on every stock daily return
for i in stocks_daily_return.columns:

  # Ignoring the date and S&P500 Columns 
  if i != 'Date' and i != '^GSPC':
    # plot a scatter plot between each individual stock and the S&P500 (Market)
    stocks_daily_return.plot(kind = 'scatter', x = '^GSPC', y = i)
    
    # Fit a polynomial between each stock and the S&P500 (Poly with order = 1 is a straight line)
    b, a = np.polyfit(stocks_daily_return['^GSPC'], stocks_daily_return[i], 1)
    
    plt.plot(stocks_daily_return['^GSPC'], b * stocks_daily_return['^GSPC'] + a, '-', color = 'r')
    plt.title("{}/SP500".format(i))
    beta[i] = b
    
    alpha[i] = a
    
    plt.show()
pprint(beta)
pprint(alpha)
ER = {}

new_stocks = list(beta.keys())

for i in new_stocks:
    ER[i] = rf + (beta[i]*(rm-rf))
    print("Valor de Retorno Esperado para {} es: {}". format(i,ER[i]))
    
portfolio_weights = 1/10 * np.ones(10) 
pprint(portfolio_weights)

ER_portfolio = sum(list(ER.values()) * portfolio_weights)
ER_portfolio

print('Expected Return Based on CAPM for the portfolio is {}%\n'.format(ER_portfolio))