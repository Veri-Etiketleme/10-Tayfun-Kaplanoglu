"""
-------------------------------------------------------
predict.py
[program description]
-------------------------------------------------------
Author:  Mohammed Perves
ID:      170143440
Email:   moha3440@mylaurier.ca
__updated__ = "2018-06-20"
-------------------------------------------------------
"""
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from random import randint
from sentiment import sentiment_analysis
import stock_data

def analysis(day, company, symbol):
    
    filename = 'finalized_model.sav'
    # load the model from disk
    loaded_model = joblib.load(filename)

    predicted_price = loaded_model.predict(day)[0]
    
    print("Sentiment Analysis")
    print("-"*20)
    num = 100
    p, n, ne = sentiment_analysis(company, 100)
    ## declare the variables for the pie chart, using the Counter variables for "sizes"
    colors = ['green', 'red', 'grey']
    sizes = [p,n,ne]
    labels = str(p)+"%", str(n)+"%", str(ne)+"%"
    
    ## use matplotlib to plot the chart
    plt.figure(figsize=(5,3))
    plt.pie(
       x=sizes,
       shadow=True,
       colors=colors,
       labels=labels,
       startangle=90)
    plt.legend(('Positive', 'Negative', 'Neutral'))
    plt.title("Sentiment of {} Tweets about {}".format(num, company))
    
    
    #style.use('fivethirtyeight')
    
    #fig = plt.figure()
    #ax1 = fig.add_subplot(1,1,1)
    #ani = animation.FuncAnimation(fig, animate(symbol), interval=1000)
    weight = 0
    randbias = randint(-5,5)
    
    print()
    print("Price Analysis")
    print("-"*20)
    latest_price = stock_data.fetch_price(symbol)
    print("Latest Price: ", latest_price)
    
    if predicted_price < latest_price:
        if p < 2 * ne:
            print("Price is LIKELY to FALL.")
        else:
            print("Price MAY FALL or STAY the SAME.")
    else:
        if p > 2 * ne:
            print("Price is LIKELY to RISE.")
        else:
            print("Price MAY RISE or STAY the SAME.")

        
    print("Price Prediction")
    print("-"*20)  
    #inversed = inverse_normalization(predicted_price)
    print("Actual Predicted Price: ", predicted_price)
    print("Predicted Price: {:.6} - {:.6}".format(predicted_price + randbias, predicted_price + weight))
    plt.show()

day = int(input("Enter the day (M/DD): "))
company = input("Company:")
symbol = input("Symbol:")

for x in range(1):
    analysis(day, company, symbol)
    
                
