#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL', 0)

for k,v in data_dict.items():
	if data_dict[k]['salary'] > 1E6 and data_dict[k]['bonus'] > 5E6: 
		print "High earner =", k, "and they make $", data_dict[k]['salary'] + data_dict[k]['bonus']

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]

    '''
    if salary > 2E7: 
    	print "High salary =", salary
    	print "High bonus =", bonus
	'''
    
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

