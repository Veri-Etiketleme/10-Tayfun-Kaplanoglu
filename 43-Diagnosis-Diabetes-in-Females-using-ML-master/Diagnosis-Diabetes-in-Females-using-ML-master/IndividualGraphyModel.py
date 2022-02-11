import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes =True)
%matplotlib inline
#
diab=pd.read_csv("diabetes.csv")
#
diab.describe()
#
diab.hist(bins=50, figsize=(20, 15))
plt.show()