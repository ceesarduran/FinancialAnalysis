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



data = yf.download(stocks, period= "5y", interval = "1d", groupby = stocks)
#print(type(data))
stocks_df = pd.DataFrame(data["Adj Close"]) 
df2 = stocks_df.reset_index(level='Date', col_level=1, col_fill='')

df2.to_excel("BuildStocks.xlsx")
df2.to_csv("BuildStocks.csv")

def interactive_plot(df, title):
    fig = px.line(title = title)
    for i in df.columns[1:]:
        fig.add_scatter(x = df['Date'], y = df[i], name = i)
    fig.show()

def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]
    return x

dfnormalized = normalize(df2)

dfnormalized.to_excel("NormalizedStocks.xlsx")

interactive_plot(df2, "StockData")
interactive_plot(dfnormalized, "StockData_Normalized")


def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[1:]:
        for j in range (1, len(df)):
            df_daily_return[i][j] = ((df[i][j] - df[i][j-1])/df[i][j-1])*100
        df_daily_return[i][0]=0
    return df_daily_return

stocks_daily_return = daily_return(df2)
stocks_daily_return.to_excel("Daily_Return.xlsx")

#stocks_daily_return.plot(kind = 'scatter' , x = '^GSPC', y = 'AAPL')
#plt.grid()
#plt.show()

beta, alpha = np.polyfit(stocks_daily_return['^GSPC'], stocks_daily_return['AAPL'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('AAPL', beta, alpha))

#charlie.plot(kind = 'scatter', x = 'SPY', y = 'AAPL')

# Straight line equation with alpha and beta parameters 
# Straight line equation is y = beta * rm + alpha
#charlie.plot(charlie['SPY'], beta * charlie['SPY'] + alpha, '-', color = 'r')
stocks_daily_return.plot(kind = 'scatter', x = '^GSPC', y = 'AAPL')
plt.plot(stocks_daily_return['^GSPC'], beta * stocks_daily_return['^GSPC'] + alpha, '-', color = 'r')

plt.grid()
plt.show()


print("Promedio de SP500 {}" .format(stocks_daily_return["^GSPC"].mean()))

rm = stocks_daily_return["^GSPC"].mean() * 252


print("Promedio de Retorno anual: {}" .format(rm))

rf = 0

ER_AAPL = rf + (beta*(rm-rf))

print("Return of Apple: {}" .format(ER_AAPL ))
#plt.scatter


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
    
    #plt.plot(stocks_daily_return['^GSPC'], b * stocks_daily_return['^GSPC'] + a, '-', color = 'r')
    #plt.title("{}/SP500".format(i))
    beta[i] = b
    
    alpha[i] = a
    
    #plt.show()
pprint(beta)
pprint(alpha)
ER = {}

new_stocks = list(beta.keys())

for i in new_stocks:
    ER[i] = rf + (beta[i]*(rm-rf))
    print("Valor de Retorno Esperado para {} es: {}".format(i,ER[i]))
    
portfolio_weights = 1/10 * np.ones(10) 
pprint(portfolio_weights)

ER_portfolio = sum(list(ER.values()) * portfolio_weights)
ER_portfolio

print('Expected Return Based on CAPM for the portfolio is {}%\n'.format(ER_portfolio))

for i in stocks_daily_return.columns:
      
  if i != 'Date' and i != '^GSPC':
    
    # Use plotly express to plot the scatter plot for every stock vs. the S&P500
    fig = px.scatter(stocks_daily_return, x = '^GSPC', y = i, title = i)

    # Fit a straight line to the data and obtain beta and alpha
    b, a = np.polyfit(stocks_daily_return['^GSPC'], stocks_daily_return[i], 1)
    
    # Plot the straight line 
    fig.add_scatter(x = stocks_daily_return['^GSPC'], y = b*stocks_daily_return['^GSPC'] + a)
    fig.show()