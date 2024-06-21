import os
import json
import pandas as pd
import numpy as np
import glob

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib
import matplotlib.pyplot as plt

raw_data_path = "../../raw"
drop_columns_list = ['review_id', 'rating', 'author']
twit = Okt()


def collect_reviews():
    all_data = []

    for filename in os.listdir(raw_data_path):
        file_path = os.path.join(raw_data_path, filename)
        print(filename, 'is processing...')
        # JSON 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        df = pd.json_normalize(data)
        df = df.drop(columns=drop_columns_list)

        # 연월 추출 (문자열 인덱싱 사용)
        df['year_month'] = df['date'].str[:5]

        all_data.append(df)

    # 모든 데이터를 하나의 데이터프레임으로 병합
    combined_df = pd.concat(all_data, ignore_index=True)

    # 연월별로 리뷰를 하나의 문자열로 합치기
    grouped = combined_df.groupby('year_month').agg({
        'review': lambda x: ' '.join(x.fillna('').astype(str)),
        'movie_id': 'first'
    }).reset_index()

    # date 열에 year_month 값을 넣기
    grouped['date'] = grouped['year_month'].str.replace('.', '-')
    grouped = grouped.drop(columns=['year_month'])

    # CSV 파일로 저장
    grouped.to_csv('combined_reviews.csv', index=False, encoding='utf-8-sig')

    return grouped


movie_review_doc = collect_reviews()
