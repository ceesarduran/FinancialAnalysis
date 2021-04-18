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

data1 = data.iloc[1:]
data1.to_excel("data.xlsx")
stock_df = stock_df.drop('Close', 1)
stock_df = stock_df.drop('Low', 1)
stock_df = stock_df.drop('Open', 1)
stock_df = stock_df.drop('High', 1)
stock_df = stock_df.drop('Volume', 1)
stock_df = stock_df.tail(-1)
print(stock_df)
#stock_df = stock_df.drop('Close', 1)
# saving the dataframe 
#stock_df.rename(columns={"" : "AAAAAAAAAAAPEPLWLD"})


stock_df.to_excel('BuildStocks.xlsx')