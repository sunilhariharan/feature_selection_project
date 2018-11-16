# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    model=RandomForestClassifier()
    rfe = RFE(model, 17)
    rfe = rfe.fit(X, y)
    r=rfe.ranking_
    l=np.where(r==1)[0].tolist()
    top_features=X.iloc[:,l].columns.values.tolist()
    return top_features


