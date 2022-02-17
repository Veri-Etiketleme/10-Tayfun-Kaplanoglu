from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

class TopQuantile(BaseEstimator, TransformerMixin):
    '''
    Engineer a new feature using the top quantile values of a given set of features. 
    
    For every value in those features, check to see if the value is within the top q-quantile
    of that feature. If so, increase the count for that sample by +1. New feature is an integer count
    of how often each sample had a value in the top q-quantile of the specified features.
    
    This class's fit(), transform(), and fit_transform() methods all assume a pandas DataFrame as input.
    '''
    
    
    def __init__(self, new_feature_name = 'top_finance', feature_list = None, q = 0.90):
        '''
        Constructor for TopQuantile objects. 
        
        Parameters
        ----------
        new_feature_name: str. Name of the feature that will be added as a pandas DataFrame column
                            upon transformation. Only used if X is a DataFrame.
        
        feature_list: list of str or int.
            If X is a Dataframe: Names of feature columns that should be included in 
                                    the count of top quantile membership.
            If X is a 2D numpy array: Integer positions for the columns to be used
                        
        q: float. Corresponds to the percentage quantile you want to be counting for. For example,
            q = 0.90 looks at the 90% percentile (top decile).
        '''
        self.new_feature_name = new_feature_name
        self.feature_list = feature_list
        self.q = q

    def fit(self, X, y = None):
        '''
        Calculates the q-quantile properly both for features that are largely positive
        and ones that are largely negative (as DataFrame.quantile() does not do this correctly).
        For example, if most of a feature's data points are between (-1E5,0), the "top decile"
        should not be -100, it should be -1E4.
        
        Parameters
        ----------
        X: features DataFrame or numpy array, one feature per column
        y: labels DataFrame/numpy array, ignored
        '''
        
        
        if isinstance(X, pd.DataFrame):
            #Is self.feature_list something other than a list of strings?
            if not isinstance(self.feature_list[0], str):
                raise TypeError('feature_list is not a list of strings')
            
            #Majority-negative features need to check df.quantile(1-q)
                #in order to be using correct quantile value
            pos = X.loc[:,self.feature_list].quantile(self.q)
            neg = X.loc[:,self.feature_list].quantile(1.0-self.q)

            #Replace negative quantile values of neg within pos to create 
            #merged Series with proper quantile values for majority-positive
            #and majority-negative features
            pos.loc[neg < 0] = neg.loc[neg < 0]
            self.quants = pos
        
        #Are features a NumPy array?
        elif isinstance(X, np.ndarray) or isinstance(X, list):
            if isinstance(X, list):
                #Need to be working with a numpy array for this to go
                #as expected
                X = np.array(X)

            #Is self.feature_list something other than a list of int?
            if not isinstance(self.feature_list[0], int):
                raise TypeError('feature_list is not a list of integers')
            
            #Majority-negative features need to check df.quantile(1-q)
                #in order to be using correct quantile value
            pos = np.percentile(X[:, self.feature_list], self.q * 100, axis = 0)
            neg = np.percentile(X[:, self.feature_list], (1.0 - self.q) * 100, axis = 0)
            
            #It's easier to work in a DataFrame, and now we don't need to know column names,
            #so let's switch over to a DataFrame for a moment
            #pos = pd.DataFrame(pos)
            #neg = pd.DataFrame(neg)
            
            #Replace negative quantile values of neg within pos to create 
            #merged Series with proper quantile values for majority-positive
            #and majority-negative features
            pos[neg < 0] = neg[neg < 0]
            self.quants = pos
        
        else:
            raise TypeError('Features need to be either pandas DataFrame or numpy array')
            
        
        

    def transform(self, X):
        '''
        Using quantile information from fit(), adds a new feature to X that contains integer counts
        of how many times a sample had a value that was in the top q-quantile of its feature, limited
        to only features in self.feature_list
        
        Parameters
        ----------
        X: features DataFrame or numpy array, one feature per column
        
        Returns
        ----------
        If X is a DataFrame: Input DataFrame with additional column for new_feature, called self.new_feature_name
        
        If X is a 2D numpy array: same as for the DataFrame case, except is a numpy array with no column names
        
        '''
        #Change all values in X to True or False if they are or are not within the
            #top q-quantile
        if isinstance(X, pd.DataFrame):
            self.boolean = X.loc[:,self.feature_list].abs() >= self.quants.abs()

            #Sum across each row to produce the counts
            X[self.new_feature_name] = self.boolean.sum(axis = 1)
            
            
        elif isinstance(X, np.ndarray) or isinstance(X, list):
            if isinstance(X, list):
                #Need to be working with a numpy array for this to go
                #as expected
                X = np.array(X)
                
            self.boolean = np.absolute(X[:,self.feature_list]) >= np.absolute(self.quants)            
            X = np.vstack((X.T, np.sum(self.boolean, axis = 1))).T
            
        else:
            raise TypeError('Features need to be either pandas DataFrame or numpy array')    
        
        return X

    def fit_transform(self, X, y = None):
        '''
        Provides the identical output to running fit() and then transform() in one nice little package.
        
        Parameters
        ----------
        X: features DataFrame or 2D numpy array, one feature per column
        y: labels DataFrame, ignored
        '''
        
        self.fit(X, y)
        return self.transform(X)