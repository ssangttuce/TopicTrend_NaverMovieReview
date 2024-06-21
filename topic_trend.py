import pandas as pd
import numpy as np

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib
import matplotlib.pyplot as plt

twit = Okt()


def tokenizer(doc):
    tokens = [word for word in twit.nouns(doc) if len(word) > 2]
    if not tokens:
        tokens = [""]
    return tokens


def test_perplexity(cv, start=5, end=15, max_iter=5, topic_word_prior=0.1, doc_topic_prior=1.0):
    print('test perplexity start...')
    iter_num = []
    per_value = []
    for i in range(start, end + 1):
        lda = LatentDirichletAllocation(n_components=i,
                                        max_iter=max_iter,
                                        topic_word_prior=topic_word_prior,
                                        doc_topic_prior=doc_topic_prior,
                                        learning_method='batch',
                                        n_jobs=-1,
                                        random_state=7)
        lda.fit(cv)
        iter_num.append(i)
        per_value.append(lda.perplexity(cv))
    # plt.plot(iter_num, per_value, 'g-')
    # plt.xlabel('Number of Topics')
    # plt.ylabel('Perplexity')
    # plt.show()
    # plt.ioff()  # 대화형 모드 끄기
    return start + per_value.index(min(per_value))


movie_review_doc = pd.read_csv('./combined_reviews.csv')
print(movie_review_doc.shape)
# 토큰화 및 벡터화
cv = CountVectorizer(tokenizer=tokenizer, max_features=2000, min_df = 2, max_df=0.4)
review_cv = cv.fit_transform(movie_review_doc['review'])

# 최적의 토픽 수 찾기
best_num_topic = test_perplexity(review_cv, start=1, end=15)
print('best_num_topic:', best_num_topic)

lda_best = LatentDirichletAllocation(n_components=best_num_topic,
                                     max_iter=5,
                                     topic_word_prior=0.1,
                                     doc_topic_prior=1.0,
                                     learning_method='batch',
                                     n_jobs=-1,
                                     random_state=7)

movie_review_topic = lda_best.fit_transform(review_cv)


def print_top_words(model, feature_names, n_top_words):
    print(model.components_.shape)
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d: " % topic_idx, end='')
        print(", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


print_top_words(lda_best, cv.get_feature_names_out(), 5)

trend_data = pd.DataFrame(movie_review_topic, columns=['Topic' + str(i) for i in range(1, best_num_topic + 1)])
trend_data = pd.concat([trend_data, movie_review_doc.date], axis=1)
trend = trend_data.groupby(['date']).mean()

fig, axes = plt.subplots(1, 3, sharey='row', figsize=(36, 12))
for col, ax in zip(trend.columns.tolist(), axes.ravel()):
    ax.set_title(col)
    ax.axes.xaxis.set_visible(False)
    ax.plot(trend[col])
plt.show()
