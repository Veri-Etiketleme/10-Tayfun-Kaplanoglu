"""
-------------------------------------------------------
model.py
[program description]
-------------------------------------------------------
Author:  Mohammed Perves
ID:      170143440
Email:   moha3440@mylaurier.ca
__updated__ = "2018-06-20"
-------------------------------------------------------
"""
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
import stock_data
#import matplotlib.animation as animation
#from matplotlib import style
#from googlefinance.client import get_price_data
#from datetime import datetime
from sklearn.externals import joblib

scaler = preprocessing.StandardScaler()
dates = []
prices = []

"""
def animate(symbol):
    graph_data = open('example.txt','r').read()
    lines = graph_data.split('\n')
    param = {
    'q': "TSLA", # Stock symbol (ex: "AAPL")
    'i': "3600", # Interval size in seconds ("86400" = 1 day intervals)
    'x': "NASD", # Stock exchange symbol on which stock is traded (ex: "NASD")
    'p': "1M" # Period (Ex: "1Y" = 1 year)
    }
    df = get_price_data(param)
    dates = df.index.tolist()
    xs = []
    for value in dates:
        xs.append(value.timestamp())
    ys = df["Close"].tolist()
    ax1.clear()
    ax1.plot(xs, ys
            
    return
"""


def normalize_data(data):
    np_data = np.array(data, dtype=float)
    np_data = np_data.reshape(-1,1)
    scaler.fit(np_data)
    normalized = scaler.transform(np_data)
    # inverse transform and print the first 5 rows
    
    return normalized

def inverse_normalization(data):
    inversed = scaler.inverse_transform(data)
    
    return inversed

def format_dates(rows):
    for row in rows:
        #date=row[:4] + row[5:7] + row[8:]
        date=row[5:7] + row[8:]
        dates.append(int(date))

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)    # skipping column names
        for row in csvFileReader:
            date = row.split('-')[1] + row[0].split('-')[2]
            dates.append(int(date))
            prices.append(float(row[4]))
    return

def train_model(dates, prices):
    dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1
    #svr_lin = SVR(kernel= 'linear', C= 1e3)
    #svr_poly = SVR(kernel= 'poly', degree= 2)
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.3) # defining the support vector regression models
    svr_rbf.fit(dates, prices) # fitting the data points in the models
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(dates, prices)
    
    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
    #plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
    #plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
    plt.plot(dates, regr.predict(dates), color='green', label= 'Linear model', linewidth=3)
    
    plt.xticks(())
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Support Vector Regression')
    plt.legend()
    
    # save the model to disk
    filename = 'finalized_model.sav'
    joblib.dump(svr_rbf, filename)
    plt.show()

    return #svr_rbf.predict(x)[0], regr.predict(x)[0]
    #svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

#get_data('TSLA_annual.csv') # calling get_data method by passing the csv file to it'

data = stock_data.fetch_data("AAPL")
prices = data['close']
format_dates(data.index.values)
#dates = normalize_data(dates)
#prices = normalize_data(prices)
#print(data.index.values)
#print(dates)
train_model(dates, prices)
print("Finished Training Model...")