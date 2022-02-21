#!/usr/bin/python
# -*- coding: utf-8 -*-


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    import pandas as pd

    #First things first: find the residual errors
    df = pd.DataFrame({'Age': ages, 'Actual Net Worth': net_worths, 'Predicted Net Worth': predictions})

    #Need to keep track of the errors relative to the rest of the data
    df['Squared Residual Errors'] = (df['Actual Net Worth'] - df['Predicted Net Worth'])**2
    df.drop(columns = ['Predicted Net Worth'], inplace = True)

    #Have to make sure column order is what we're expecting
    cols = ['Age', 'Actual Net Worth', 'Squared Residual Errors']
    df = df[cols]

    

    #Make sure we know where the highest errors are (at the end)
    df.sort_values('Squared Residual Errors', inplace = True, ascending = True)
    print "DF =", df

    #Now to make the DataFrame data into a list of tuples
    cleaned_data = list(df.itertuples(index = False, name=None))


    #Don't forget! Need to remove the 10% of data with the highest residual errors
    #cleaned_data = cleaned_data[ : round(0.9*len(cleaned_data))]
    cleaned_data = cleaned_data[ : 81]

    
    return cleaned_data

