import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,StandardScaler
from scipy.stats import pearsonr
import pandas as pd

import sklearn.metrics as metrics

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
def preprocess(df):
    raw_df = df.copy()
    raw_df = raw_df[pd.to_numeric(raw_df['월'], errors='coerce').notnull()].astype({'월': int})
    raw_df = raw_df[pd.to_numeric(raw_df['년'], errors='coerce').notnull()].astype({'년': int})
    raw_df = raw_df[pd.to_numeric(raw_df['일'], errors='coerce').notnull()].astype({'일': int})
    raw_df['계약일'] = pd.to_datetime(raw_df['년'].astype(str) + '-' + raw_df['월'].astype(str) + '-'+ raw_df['일'].astype(str))    
    raw_df['전용면적'] = pd.to_numeric(raw_df['전용면적'],errors = 'coerce')
    raw_df['거래금액'] = raw_df['거래금액'].apply(lambda x : int(x.replace(',','')))
    raw_df['건축년도'] = pd.to_numeric(raw_df['건축년도'],errors = 'coerce')
    raw_df['건축년도'] = raw_df['년'] - raw_df['건축년도']
    raw_df['중개사소재지'] = raw_df['중개사소재지'].apply(lambda x : '미확인' if x == ' ' else x)
    raw_df['거래유형'] = raw_df['거래유형'].apply(lambda x : '미확인' if x == ' ' else x)
    raw_df['법정동'] = raw_df['법정동'].apply(lambda x : x.strip())
    del raw_df['Unnamed: 0']
    del raw_df['해제사유발생일']
    del raw_df['해제여부']
    del raw_df['거래유형']
    # del raw_df['년']
    # del raw_df['월']
    # del raw_df['일']
    del raw_df['중개사소재지']
    del raw_df['지역코드']
    del raw_df['지번']
    return raw_df
# 기본 데이터 전처리
# def preprocess(df):
#     raw_df = df.copy()
#     raw_df['년'] = df['계약일'][0].year
#     raw_df['월'] = df['계약일'][0].month
#     raw_df['일'] = df['계약일'][0].day
#     raw_df['전용면적'] = pd.to_numeric(raw_df['전용면적'],errors = 'coerce')
#     raw_df['거래금액'] = raw_df['거래금액'].apply(lambda x : int(x.replace(',','')))
#     raw_df['법정동'] = raw_df['법정동'].apply(lambda x : x.strip())
#     return raw_df
def economic_merge_data(df,columns=None):
    tmp = df.copy()
    print('기준금리_start')
    geumri = pd.read_csv('전처리_완료/기준금리.csv')
    geumri['년'] = geumri['변환'].apply(lambda x : int(str(x)[:4]))
    geumri['월'] = geumri['변환'].apply(lambda x : int(str(x)[4:]))
    tmp = pd.merge(tmp, geumri, how='left', left_on=['년','월'], right_on=['년','월']).drop(['변환'], axis=1)
    tmp = tmp.rename(columns = {'원자료' : '기준금리'})
    print('기준금리_end')
    print('GDP_start')
    gdp = pd.read_csv('전처리_완료/gdp.csv')
    gdp['년'] = gdp['날짜'].apply(lambda x : int(x.split('-')[0]))
    gdp['월'] = gdp['날짜'].apply(lambda x : int(x.split('-')[1]))
    tmp = pd.merge(tmp, gdp, how='left', left_on=['년','월'], right_on=['년','월']).drop(['날짜'],axis=1)
    tmp['국내총생산(명목GDP)'] = tmp['국내총생산(명목GDP)'].apply(lambda x : float(x.replace(',', '')))
    print('GDP_end')
    if columns is not None:
        columns.extend(['국내총생산(명목GDP)','경제성장률(실질GDP성장률)','기준금리'])
    else:
        columns = ['국내총생산(명목GDP)','경제성장률(실질GDP성장률)','기준금리']
    return tmp,columns
def social_merge_data(df,columns=None):
    tmp = df.copy()
    print('공원수_start')
    park = pd.read_csv('전처리_완료/강남구공원수자료.csv')
    park['공원수'] = park['공원수'].astype(int)
    tmp = pd.merge(tmp, park, how='left', left_on='법정동', right_on='법정동')
    print('공원수_end')
    print('인구밀도_start')
    humane_rate = pd.read_csv('전처리_완료/인구밀도.csv')
    humane_rate['시점'].astype(int)
    humane_rate = humane_rate.melt(id_vars='시점', var_name='법정동', value_name='인구밀도')
    def mapped_dong(example):
        items = df['법정동'].unique()
        for dong in items:
            dong = dong.strip()
            if len(dong) == 3:
                if example.startswith(dong[:2]):
                    return dong
            else:
                if example.startswith(dong[:3]):
                    return dong
        return ''
    humane_rate['법정동'] = humane_rate['법정동'].apply(mapped_dong)
    humane_rate['인구밀도'] = humane_rate['인구밀도'].apply(lambda x: 0 if x == '-' else int(x))
    humane_rate = humane_rate.groupby(['시점','법정동']).mean().reset_index()
    tmp = pd.merge(tmp, humane_rate, how='left', left_on=['년','법정동'], right_on=['시점','법정동']).drop('시점',axis=1)
    print('인구밀도_end')
    if columns is not None:
        columns.extend(['인구밀도','공원수'])
    else:
        columns = ['인구밀도','공원수']
    return tmp,columns
def bodong_merge_data(df,columns=None):
    tmp = df.copy()
    print('선도50지수_start')
    top50_rate= pd.read_csv('전처리_완료/선도아파트50지수.csv')
    top50_rate['년'] = top50_rate['날짜'].apply(lambda x : int(x.split('-')[0]))
    top50_rate['월'] = top50_rate['날짜'].apply(lambda x : int(x.split('-')[1]))
    tmp = pd.merge(tmp, top50_rate, how='left', left_on=['년','월'], right_on=['년','월']).drop(['날짜','전월대비증감률'],axis=1)
    print('선도50지수_end')
    # 전월 대비만 빼주십사~ - 감소 : 스케일링 잘못하면
    
    print('전세가격지수_start')
    jeonse = pd.read_csv('전처리_완료/전세가격지수.csv')
    tt = jeonse.transpose()
    tt.columns = tt.iloc[0].tolist()
    tt = tt['강남구']
    tt = pd.DataFrame(tt.iloc[1:])
    year = pd.Series(list(map(lambda x : str(x)[:4],tt.index)))
    month = pd.Series(list(map(lambda x : str(x)[4:],tt.index)))
    tt.reset_index(inplace=True)
    tt['년'] = year.astype(int)
    tt['월'] = month.astype(int)
    tmp = pd.merge(tmp, tt, how='left', left_on=['년','월'], right_on=['년','월']).drop(['index'],axis=1)
    tmp = tmp.rename(columns = {'강남구' : '전세가격지수'})
    print('전세가격지수_end')
    print('매매가격지수_start')
    gangnam = pd.read_csv('전처리_완료/강남구_12_22가격지수.csv')
    gangnam['년'] = gangnam['날짜'].apply(lambda x : int(x.split('-')[0]))
    gangnam['월'] = gangnam['날짜'].apply(lambda x : int(x.split('-')[1]))
    tmp = pd.merge(tmp, gangnam, how='left', left_on=['년','월'], right_on=['년','월']).drop(['날짜','지역명'],axis=1)
    tmp = tmp.rename(columns = {'가격지수' : '매매가격지수'})
    print('매매가격지수_end')
    if columns is not None:
        columns.extend(['전세가격지수','매매가격지수','선도50지수','전년동월대비증감률'])
    else:
        columns = ['전세가격지수','매매가격지수','선도50지수','전년동월대비증감률']
    return tmp,columns

def onehotencoding(tmp):
    df = tmp.copy()
    df_encoded = pd.get_dummies(df, columns=['법정동'])
    return df_encoded

def scaling(tmp,columns=None):
    df = tmp.copy()
    print(df.columns)
    # Datetime 필드를 Epoch 시간으로 변환
    df['계약일'] = df['계약일'].apply(lambda x: x.timestamp())
    # df['국내총생산(명목GDP)'] = df['국내총생산(명목GDP)'].apply(lambda x : float(x.replace(',', '')))
    
    # 수치형 데이터 스케일링 (Min-Max Scaling)
    scaler = MinMaxScaler()
    scaler2 = StandardScaler()
    # 건축년도, 거래금액, 전용면적, 층 , 계약일 스케일링 작업
    df[['건축년도','계약일', '전용면적', '층']] = scaler.fit_transform(df[['건축년도', '계약일','전용면적', '층']])
    # df[['거래금액']] = scaler2.fit_transform(df[['거래금액']])
    if columns is not None:
        df[columns] = scaler2.fit_transform(df[columns])
        print('columns' , columns)
    return df,scaler,scaler2

def get_x_y(scaled_df):
    # 종속변수 Y -> 거래금액
    scaled_df = scaled_df.dropna().reset_index()
    # Y = scaled_df['거래금액']
    Y = np.log1p(scaled_df['거래금액'])

    # 독립변수 X -> df에서 거래금액을 제외한 나머지 속성들
    X = scaled_df.drop('거래금액',axis=1)
    # except_column = ['년','월','일','계약일']
    except_column = ['년','월','일','계약일','아파트']
    for i in except_column:
        del X[i]
    del X['index']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
