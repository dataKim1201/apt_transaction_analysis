import pandas as pd
import os

def economic_merge_data(df,columns=None):
    raw_geumri =pd.read_csv('./전처리_완료/기준금리.csv')
    raw_gdp = pd.read_csv('./전처리_완료/gdp.csv')
    tmp = df.copy()
    geumri = raw_geumri.copy()
    gdp = raw_gdp.copy()
    print(os.getcwd())
    print('기준금리_start')
    
    geumri['년'] = geumri['변환'].apply(lambda x : int(str(x)[:4]))
    geumri['월'] = geumri['변환'].apply(lambda x : int(str(x)[4:]))
    tmp = pd.merge(tmp, geumri, how='left', left_on=['년','월'], right_on=['년','월']).drop(['변환'], axis=1)
    tmp = tmp.rename(columns = {'원자료' : '기준금리'})
    print('기준금리_end')
    print('GDP_start')
    
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
    raw_park = pd.read_csv('./전처리_완료/강남구공원수자료.csv')
    raw_humane_rate = pd.read_csv('./전처리_완료/인구밀도.csv')
    humane_rate, park = raw_humane_rate.copy(), raw_park.copy()
    tmp = df.copy()
    print('공원수_start')
    
    park['공원수'] = park['공원수'].astype(int)
    tmp = pd.merge(tmp, park, how='left', left_on='법정동', right_on='법정동')
    print('공원수_end')
    print('인구밀도_start')
    
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
    raw_jeonse = pd.read_csv('./전처리_완료/전세가격지수.csv')
    raw_top50_rate= pd.read_csv('./전처리_완료/선도아파트50지수.csv')
    raw_gangnam = pd.read_csv('./전처리_완료/강남구_12_22가격지수.csv')
    jeonse,gangnam, top50_rate = raw_jeonse.copy(),raw_gangnam.copy(),raw_top50_rate.copy()
    tmp = df.copy()
    print('선도50지수_start')
    
    top50_rate['년'] = top50_rate['날짜'].apply(lambda x : int(x.split('-')[0]))
    top50_rate['월'] = top50_rate['날짜'].apply(lambda x : int(x.split('-')[1]))
    tmp = pd.merge(tmp, top50_rate, how='left', left_on=['년','월'], right_on=['년','월']).drop(['날짜','전월대비증감률'],axis=1)
    print('선도50지수_end')
    # 전월 대비만 빼주십사~ - 감소 : 스케일링 잘못하면
    
    print('전세가격지수_start')

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