import pickle
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import joblib
from utils import *
def regression_results(y_true, y_pred,verbose=False):
    # Regression metrics
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    r2=metrics.r2_score(y_true, y_pred)
    if verbose:
        print('r2: ', round(r2,4))             
        print('MAE: ', round(mean_absolute_error,4))
        print('MSE: ', round(mse,4))
        print('RMSE: ', round(np.sqrt(mse),4))
    return round(np.sqrt(mse),4),r2

if __name__=='__main__':
    df = pd.read_csv('전처리_완료/12_22아파트매매.csv')
    ultra = onehotencoding(scaling(*bodong_merge_data(*social_merge_data(*economic_merge_data(df)))))
    X_train, X_test, y_train, y_test = get_x_y(ultra)
    tscv = TimeSeriesSplit(n_splits=3)
    model = RandomForestRegressor(n_estimators = 100)
    tscv = TimeSeriesSplit(n_splits=3)
    for i, (train_index, test_index) in enumerate(tscv.split(X_train)):
        train,valid = X_train.iloc[train_index,:],X_train.iloc[test_index,:]
        train_y,valid_y = y_train.iloc[train_index],y_train.iloc[test_index]
        model.fit(train,train_y)
        y_pred = model.predict(valid)
        print(f'{i + 1} result RandomForest predict result with using')
        rmse,r2 = regression_results(valid_y,y_pred)
    joblib.dump(model,'rf_model.pkl')
    # joblib.load('rf_model.pkl') 

