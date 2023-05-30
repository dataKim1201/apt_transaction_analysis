import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from scipy.stats import pearsonr
import os
import sklearn.metrics as metrics



# 기본 데이터 전처리
def preprocess(df):
    '''
    df : 기본 데이터가 되는 12년 부터 22년까지 수집된 아파트 실거래가 수집자료입니다.

    '''
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


def onehotencoding(tmp):
    df = tmp.copy()
    df_encoded = pd.get_dummies(df, columns=['법정동'],dtype=int)
    return df_encoded

def scaling(tmp,columns=None):
    df = tmp.copy()
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
    return df

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
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    del X_train['index']
    del X_test['index']

    return X_train, X_test, y_train, y_test