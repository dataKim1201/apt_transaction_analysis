import streamlit as st
import joblib
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd
from utils import *
def scaling(sample):
    sample,columns = bodong_merge_data(*social_merge_data(*economic_merge_data(sample)))
    sample['계약일'] = sample['계약일'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
    sample[['건축년도','계약일', '전용면적', '층']] = mm_scaler.transform(sample[['건축년도', '계약일','전용면적', '층']])
    # columns = ['거래금액', '건축년도', '년', '법정동', '아파트', '월', '일', '전용면적', '층', '계약일', '기준금리',
    #    '국내총생산(명목GDP)', '경제성장률(실질GDP성장률)', '공원수', '인구밀도', '선도50지수', '전년동월대비증감률',
    #    '전세가격지수', '매매가격지수']
    sample[columns] = st_scaler.transform(sample[columns])
    return sample
def one_hot(sample):
    except_column = ['년','월','일','계약일','아파트','거래금액']
    for i in except_column:
        del sample[i]

    # 법정동 리스트를 컬럼으로 추가
    for col_name in legal_dongs:
        sample['법정동_' + col_name] = (sample['법정동'] == col_name).astype(int)
    del sample['법정동']
    return sample
def get_current():
    current_time = datetime.now()
    current_date = current_time.strftime("%Y-%m-%d")

    # 1년 전 날짜 계산
    one_year_before = current_time - timedelta(days=365)
    one_year_before_date = one_year_before.strftime("%Y-%m-%d")
    return one_year_before_date
def run_inference(sample):
        sample = scaling(sample)
        sample = one_hot(sample)
        return round(np.exp(model.predict(sample[['건축년도', '전용면적', '층', '기준금리', '국내총생산(명목GDP)', '경제성장률(실질GDP성장률)', '공원수',
        '인구밀도', '선도50지수', '전년동월대비증감률', '전세가격지수', '매매가격지수', '법정동_개포동', '법정동_논현동',
        '법정동_대치동', '법정동_도곡동', '법정동_삼성동', '법정동_세곡동', '법정동_수서동', '법정동_신사동',
        '법정동_압구정동', '법정동_역삼동', '법정동_율현동', '법정동_일원동', '법정동_자곡동', '법정동_청담동']]))[0],2)


def main():
    st.title("저기 도로 건너편 아파트 가격은 얼마일까 가격은 얼마일까?")
    # 사용자 입력 받기
    # 건축년도
    build_year = st.slider("건축년도를 선택하세요", min_value=0, max_value=20, step=1) 
    # 법정동
    legal_dong = st.selectbox("원하는 동을 입력하세요.", legal_dongs)
    # 아파트
    apt_selected = st.selectbox("아파트를 입력하세요.", apt)
    # 전용면적
    area = st.slider("Select the exclusive area", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
    # 층
    floor = random.randint(1, 10)
    # 년 월 일 

    sample = {'건축년도' : [build_year],'년' : [year],'법정동' : [legal_dong],
            '아파트' : [apt_selected],'월' : [month],'일' : [day],
            '전용면적' : [area],'층' :[floor],
            '계약일' : [one_year_before_date],
            '거래금액' : [2000]}
    sample = pd.DataFrame(sample)
    if st.button("추론"):
        # 입력 데이터를 추론 함수에 전달하여 결과 출력
        result = run_inference(sample)
        st.markdown("<div style='text-align: center; font-size: 24px;'>예상 가격: {:.2f} 만원 </div>".format(result), unsafe_allow_html=True)

if __name__ == '__main__':
    legal_dongs = ['역삼동',
    '개포동',
    '청담동',
    '삼성동',
    '대치동',
    '신사동',
    '논현동',
    '압구정동',
    '세곡동',
    '일원동',
    '수서동',
    '도곡동',
    '자곡동',
    '율현동']
    apt = joblib.load('아파트.pkl')
    # 모델 로드
    model = joblib.load('rf_model.pkl')
    mm_scaler = joblib.load('mm_scaler.pkl')
    st_scaler = joblib.load('st_scaler.pkl')
    one_year_before_date = get_current()

    year = datetime.strptime(one_year_before_date,'%Y-%m-%d').year
    month = datetime.strptime(one_year_before_date,'%Y-%m-%d').month
    day = datetime.strptime(one_year_before_date,'%Y-%m-%d').day
    # Streamlit 앱 설정
    main()
