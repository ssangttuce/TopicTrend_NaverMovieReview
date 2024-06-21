# 네이버 영화 리뷰 데이터를 활용한 트픽 트렌트 분석

약 14,000편 가량의 영화에 대한 리뷰 데이터를 토피 모델링 후, 시간대별로 구분하여 토픽 트렌드 분석을 진행함

## 토픽 모델링 및 토픽 수 결정
<img width="389" alt="실행 로그" src="https://github.com/ssangttuce/TopicTrend_NaverMovieReview/assets/88099593/1e04108e-eb11-4be8-8edf-938fc6e31d0e">

## 토픽 트렌드 그래프
![토픽 트렌드 차트](https://github.com/ssangttuce/TopicTrend_NaverMovieReview/assets/88099593/3cf61f83-c9fa-4b12-aa4d-ef995a68cd20)

## 프로젝트 결과

> 부적절한 토픽의 수와 고유명사 위주의 키워드 추출 

본래 프로젝트의 목적은 영화 자체와 상관없이 리뷰가 영화 구성 요소의 어떤 부분에 집중해서 작성되었는지 시간에 따른 변화를 확인하는 것이다. 
영화의 연출, 음악, 액션, 스토리, 배우의 연기 등 영화를 이루는 다양한 요소에 대한 평가를 리뷰에서 찾아내어 영화를 보는 대중의 시선이 어떻게 변화하였는지 파악하고자 했다. 
프로젝트 진행 결과는 목적을 달성하기에 부족한 내용이다. 영화의 구성 요소를 평가하는 내용은 드러나지 않고, 추정된 토픽의 수도 너무 적어 토픽의 변화 추이를 파악할 수 없다.

## 프로젝트 문제 분석

> 데이터에 대한 이해 부족과 미흡한 자료 전처리 과정 

토픽 모델링을 통해 도출된 키워드는 영화의 배우나 고유 명사 위주로 구성되었다. 이는 리뷰 내용들을 시간대별로 병합하기 이전에 각 영화에 대한 리뷰들을 전처리했어야 한다고 생각한다. 
영화 리뷰는 시청한 영화에 대해 일반적으로 평가한 내용이 아니기 때문에 시간대별로 병합하여 전처리하기 전에 영화 별로 전처리가 필요하다.