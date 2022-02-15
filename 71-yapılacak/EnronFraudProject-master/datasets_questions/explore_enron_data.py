#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#for name in enron_data.keys():
#	if "LAY" in name: print name
#	elif "FASTOW" in name: print name

#print enron_data['LAY KENNETH L'].keys()

counter = 0
i = 0
for k,v in enron_data.items():
	if enron_data[k]['total_payments'] == 'NaN': counter += 1
	if enron_data[k]['poi']: i += 1


print counter

totalNum = len(enron_data)
print "Total Number of POIs = ", i

print "Percent = ", (counter/i)