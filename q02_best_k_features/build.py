# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k=20):
    X=data.iloc[:,:-1]    
    y=data.iloc[:,-1]
    select=SelectPercentile(f_regression, percentile=k)
    X_new=select.fit_transform(X,y)
    names=X.columns.values[select.get_support()]
    scores=select.scores_[select.get_support()]
    l=list(zip(names,scores))
    df=pd.DataFrame(data=l)
    df=df.sort_values([1],ascending=False)
    imp_features=df.iloc[:,0].values.tolist()
    
    return imp_features



