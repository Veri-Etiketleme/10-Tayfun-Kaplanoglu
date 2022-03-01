from iexfinance import Stock
from iexfinance import get_historical_data
from datetime import datetime
import matplotlib.pyplot as plt

today=datetime.today().strftime('%Y-%m-%d')
bday=datetime(2017,1,1)

def fetch_price(ticker):
    stock = Stock(ticker.upper())
    price = stock.get_price()
    return price

def fetch_data(ticker, start=bday, end=today):
    df = get_historical_data(ticker.upper(), start, end, output_format='pandas')
    #df.head()
    return df


"""ticker = input("Symbol: ").upper()

start = input("Start date (YYYY/M/D): ")
if start != "":
    start = datetime(int(start[:4]), int(start[5:6]), int(start[7:]))
else:
    start = datetime(2018,6,5)

end = input("End date (YYYY/M/D): ")
if end != "":
    end = datetime(int(end[:4]), int(end[5:6]), int(end[7:]))
else:
    end = today

print(stock_price(ticker))
print(ticker)
data = stock_data(ticker, start)

print(data)
plt.title('Time series chart for ' + ticker)
plt.plot(data['close'])
#plt.bar(data.index, data["close"])
plt.show()
"""