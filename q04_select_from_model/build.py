# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):
    np.random.seed(9)
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    clf = RandomForestClassifier()
    clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    ar=model.get_support()
    l=np.where(ar==True)[0].tolist()
    feature_name=X.iloc[:,l].columns.tolist()
    return feature_name


