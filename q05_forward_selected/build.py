# %load q05_forward_selected/build.py
# Default imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()


#Your solution code here
def forward_selected(data,model):
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    col_list=X.columns.tolist()
    r=[]
    
    r2=[]
    fe=[]
    for i in range(len(col_list)):
        feature_list=[]
        curr_feature_name=col_list[i]
        feature_list.append(curr_feature_name)
        model=LinearRegression()
        model.fit(X.loc[:,feature_list],y)
        y_pred=model.predict(X.loc[:,feature_list])
        s=r2_score(y,y_pred)
        r2.append(s)
        fe.append(feature_list)
        d=dict(zip(r2,fe))
    r21=max(r2)
    fe1=d[max(r2)]
    r.append(r21)
    
    r2=[]
    fe=[]
    for i in range(len(col_list)):
        feature_list=['OverallQual']
        curr_feature_name=col_list[i]
        feature_list.append(curr_feature_name)
        model=LinearRegression()
        model.fit(X.loc[:,feature_list],y)
        y_pred=model.predict(X.loc[:,feature_list])
        s=r2_score(y,y_pred)
        r2.append(s)
        fe.append(feature_list)
        d=dict(zip(r2,fe))
    r21=max(r2)
    fe1=d[max(r2)]
    r.append(r21)
    
    r2=[]
    fe=[]
    for i in range(len(col_list)):
        feature_list=['OverallQual','GrLivArea']
        curr_feature_name=col_list[i]
        feature_list.append(curr_feature_name)
        model=LinearRegression()
        model.fit(X.loc[:,feature_list],y)
        y_pred=model.predict(X.loc[:,feature_list])
        s=r2_score(y,y_pred)
        r2.append(s)
        fe.append(feature_list)
        d=dict(zip(r2,fe))
    r22=max(r2)
    fe2=d[max(r2)]
    r.append(r22)
    
    r2=[]
    fe=[]
    for i in range(len(col_list)):
        feature_list=['OverallQual','GrLivArea','BsmtFinSF1']
        curr_feature_name=col_list[i]
        feature_list.append(curr_feature_name)
        model=LinearRegression()
        model.fit(X.loc[:,feature_list],y)
        y_pred=model.predict(X.loc[:,feature_list])
        s=r2_score(y,y_pred)
        r2.append(s)
        fe.append(feature_list)
        d=dict(zip(r2,fe))
    r23=max(r2)
    fe3=d[max(r2)]
    r.append(r23)
    
    r2=[]
    fe=[]
    for i in range(len(col_list)):
        feature_list=['OverallQual','GrLivArea','BsmtFinSF1','GarageCars']
        curr_feature_name=col_list[i]
        feature_list.append(curr_feature_name)
        model=LinearRegression()
        model.fit(X.loc[:,feature_list],y)
        y_pred=model.predict(X.loc[:,feature_list])
        s=r2_score(y,y_pred)
        r2.append(s)
        fe.append(feature_list)
        d=dict(zip(r2,fe))
    r24=max(r2)
    fe4=d[max(r2)]
    r.append(r24)
    
    r2=[]
    fe=[]
    for i in range(len(col_list)):
        feature_list=['OverallQual','GrLivArea','BsmtFinSF1','GarageCars','KitchenAbvGr']
        curr_feature_name=col_list[i]
        feature_list.append(curr_feature_name)
        model=LinearRegression()
        model.fit(X.loc[:,feature_list],y)
        y_pred=model.predict(X.loc[:,feature_list])
        s=r2_score(y,y_pred)
        r2.append(s)
        fe.append(feature_list)
        d=dict(zip(r2,fe))
    r25=max(r2)
    fe5=d[max(r2)]
    r.append(r25)
    
    r2=[]
    fe=[]
    for i in range(len(col_list)):
        feature_list=['OverallQual','GrLivArea','BsmtFinSF1','GarageCars','KitchenAbvGr','1stFlrSF']
        curr_feature_name=col_list[i]
        feature_list.append(curr_feature_name)
        model=LinearRegression()
        model.fit(X.loc[:,feature_list],y)
        y_pred=model.predict(X.loc[:,feature_list])
        s=r2_score(y,y_pred)
        r2.append(s)
        fe.append(feature_list)
        d=dict(zip(r2,fe))
    r26=max(r2)
    fe6=d[max(r2)]
    r.append(r26)
    
    r2=[]
    fe=[]
    for i in range(len(col_list)):
        feature_list=['OverallQual','GrLivArea','BsmtFinSF1','GarageCars','KitchenAbvGr','1stFlrSF','YearRemodAdd']
        curr_feature_name=col_list[i]
        feature_list.append(curr_feature_name)
        model=LinearRegression()
        model.fit(X.loc[:,feature_list],y)
        y_pred=model.predict(X.loc[:,feature_list])
        s=r2_score(y,y_pred)
        r2.append(s)
        fe.append(feature_list)
        d=dict(zip(r2,fe))
    r27=max(r2)
    fe7=d[max(r2)]
    r.append(r27)
    
    r2=[]
    fe=[]
    for i in range(len(col_list)):
        feature_list=['OverallQual','GrLivArea','BsmtFinSF1','GarageCars','KitchenAbvGr','1stFlrSF','YearRemodAdd','LotArea']
        curr_feature_name=col_list[i]
        feature_list.append(curr_feature_name)
        model=LinearRegression()
        model.fit(X.loc[:,feature_list],y)
        y_pred=model.predict(X.loc[:,feature_list])
        s=r2_score(y,y_pred)
        r2.append(s)
        fe.append(feature_list)
        d=dict(zip(r2,fe))
    r28=max(r2)
    fe8=d[max(r2)]
    r.append(r28)
    
    r2=[]
    fe=[]
    for i in range(len(col_list)):
        feature_list=['OverallQual','GrLivArea','BsmtFinSF1','GarageCars','KitchenAbvGr','1stFlrSF','YearRemodAdd','LotArea','MasVnrArea']
        curr_feature_name=col_list[i]
        feature_list.append(curr_feature_name)
        model=LinearRegression()
        model.fit(X.loc[:,feature_list],y)
        y_pred=model.predict(X.loc[:,feature_list])
        s=r2_score(y,y_pred)
        r2.append(s)
        fe.append(feature_list)
        d=dict(zip(r2,fe))
    r29=max(r2)
    fe9=d[max(r2)]
    r.append(r29)
    
  
    
    return fe9,r


