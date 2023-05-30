from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics

def main(X_train, X_test, y_train, y_test,test_name):
    np.random.seed(42)
    # Spot Check Algorithms
    models = []
    # models.append(('LR', LinearRegression()))
    models.append(('SVR', SVR(kernel = 'linear',C=2.0,gamma='scale'))) # kernel = linear
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('NN', MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.01, max_iter=50, batch_size=64, shuffle=True, random_state=777)))  #neural network
    models.append(('HGB', HistGradientBoostingRegressor())) 
    models.append(('RF', RandomForestRegressor(n_estimators = 100))) 
    # Ensemble method - collection of many decision trees
    # Evaluate each model in turn
    results = []
    names = []
    test_results = []
    for name, model in models:
        # TimeSeries Cross validation
        tmp = []
        tscv = TimeSeriesSplit(n_splits=3)
        for i, (train_index, test_index) in enumerate(tscv.split(X_train)):
            train,valid = X_train.iloc[train_index,:],X_train.iloc[test_index,:]
            train_y,valid_y = y_train.iloc[train_index],y_train.iloc[test_index]
            model.fit(train,train_y)
            y_pred = model.predict(valid)
            print(f'{i + 1} result {test_name} predict result with using {name}')
            rmse,r2 = regression_results(valid_y,y_pred)
            tmp.append(r2)
        # cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
        y_pred = model.predict(X_test)
        test_result = regression_results(y_test,y_pred,True)
        if test_result[1] <0:
            test_results.append(0)
        else:
            test_results.append(test_result[1])
        results.append(tmp)
        names.append(name)
        print('%s: %f (%f)' % (name, np.array(tmp).mean(), np.array(tmp).std()))
    # print(f'maximum test result is {max(test_results)} and its model is {names[test_results.index(max(test_results))]}')
    # Compare Algorithms
    # plt.bar(names, height=test_results,width=0.6,)
    plt.boxplot(results, labels=names)
    plt.title(f'{test_name} R2 score Algorithm Comparison')
    plt.show()

def regression_results(y_true, y_pred,verbose=False):
    '''
    평가지표를 위한 코드입니다. 
    y_true : 실제 라벨에 대한 값으로 거래금액의 log변환을 취한 결과입니다.
    y_pred : 모델이 예측한 값으로 실제 값과 얼마나 다른지 비교하기 위한 값입니다.
    verbose : 출력 여부를 갖는 불리언 값으로 True시 결과값을 출력합니다.

    sklearn metric을 활용하여  R2, MAE, MSE, RMSE에 대한 정보를 반환합니다.
    '''
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
