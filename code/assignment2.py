import numpy as np
import pandas as pd
from nltk import RegexpTokenizer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from MNB import MNB


def module_analysis():
    mnb = MNB()
    data = mnb.load_data()
    token = RegexpTokenizer(r'[a-zA-Z0-9]{2,}')  # only consider alphabet and number
    cv = CountVectorizer(ngram_range=(1, 1), tokenizer=token.tokenize, min_df=2)  # stop_words=ENGLISH_STOP_WORDS,
    X_train, X_test, Y_train, Y_test = train_test_split(data[:, 0], data[:, 1], test_size=0.2, random_state=25)
    cv.fit(X_train)
    neg_train = X_train[Y_train == 'neg']
    pos_train = X_train[Y_train == 'pos']
    neg_train = cv.transform(neg_train)
    pos_train = cv.transform(pos_train)

    neg_tf = np.sum(neg_train, axis=0)
    pos_tf = np.sum(pos_train, axis=0)
    neg_tf = np.squeeze(np.asarray(neg_tf))
    pos_tf = np.squeeze(np.asarray(pos_tf))
    data_frame = pd.DataFrame([neg_tf, pos_tf], columns=cv.get_feature_names()).transpose()
    data_frame.columns = ['negative', 'positive']
    neg_tf = data_frame.sort_values(by=["negative"], ascending=False).iloc[:100]  # to see the stop words effect
    pos_tf = data_frame.sort_values(by=["positive"], ascending=False).iloc[:100]  # to see the stop words effect

    # this part is to show whose presence most strongly predicts that the review is positive
    # and whose absence most strongly predicts that the review is positive.
    data_frame['pos_rate'] = data_frame['positive'] * 1. / (data_frame['positive'] + data_frame['negative'])
    pos_rate = data_frame.sort_values(by=["pos_rate", "positive"], ascending=[False, False])  # for presenting positive
    positive_common = data_frame[data_frame["positive"] > 200].sort_values(by=["pos_rate"], ascending=False)

    data_frame['neg_rate'] = data_frame['negative'] * 1. / (data_frame['positive'] + data_frame['negative'])
    neg_rate = data_frame.sort_values(by=["neg_rate", "negative"], ascending=[False, False])  # for presenting negative
    negative_common = data_frame[data_frame["negative"] > 200].sort_values(by=["neg_rate"], ascending=False)

    print("\nList the 10 words whose presence most strongly predicts that the review is positive ")
    print(pos_rate.iloc[:20])
    print("\nList the 10 words whose absence most strongly predicts that the review is positive.")
    print(negative_common.iloc[:20])
    print("\nList the 10 words whose presence most strongly predicts that the review is negative.")
    print(neg_rate.iloc[:20])
    print("\nList the 10 words whose absence most strongly predicts that the review is negative")
    print(positive_common.iloc[:20])

    print("\nStopwords are everywhere and they are on the both negative and positive sentences")
    print(neg_tf)
    print()
    print(pos_tf)


def mnb_score_calculator(ngram, six_cat=False, stop_w=None):
    token = RegexpTokenizer(r'[a-zA-Z0-9]{2,}')  # only consider alphabet and number
    cv = CountVectorizer(stop_words=stop_w, ngram_range=ngram, tokenizer=token.tokenize,
                         min_df=2)  # stop_words=ENGLISH_STOP_WORDS,
    # min_df was used for ignoring the words which was encountered less than 2

    mnb = MNB()
    data = mnb.load_data(six_category=six_cat)  # first load the data_set

    text_counts = cv.fit_transform(data[:, 0])  # column zero means data_set and column 1 is label
    X_train, X_test, Y_train, Y_test = train_test_split(text_counts, data[:, 1], test_size=0.2, random_state=25)

    mnb.fit(X_train, Y_train)
    predicted = mnb.predict(X_test.toarray())
    return mnb.accuracy_calculator(predicted, Y_test)


unigram = (1, 1)
bigram = (2, 2)
unigram_bigram = (1, 2)

###
###Without stopwords
###
# Positive negative classifier
print(
    "Accuracy without stopwords of unigram is {0:.3f}".format(mnb_score_calculator(unigram, stop_w=ENGLISH_STOP_WORDS)))
print("Accuracy without stopwords of bigram is {0:.3f}".format(mnb_score_calculator(bigram, stop_w=ENGLISH_STOP_WORDS)))
print("Accuracy without stopwords of unigram_bigram is {0:.3f}".format(
    mnb_score_calculator(unigram_bigram, stop_w=ENGLISH_STOP_WORDS)))

# Six category classifier
print("Accuracy without stopwords of six-category unigram is {0:.3f}".format(
    mnb_score_calculator(unigram, six_cat=True, stop_w=ENGLISH_STOP_WORDS)))
print("Accuracy without stopwords of six-category bigram is {0:.3f}".format(
    mnb_score_calculator(bigram, six_cat=True, stop_w=ENGLISH_STOP_WORDS)))
print("Accuracy without stopwords of six-category unigram_bigram is {0:.3f}".format(
    mnb_score_calculator(unigram_bigram, six_cat=True, stop_w=ENGLISH_STOP_WORDS)))

###
### With stopwords
###

print("Accuracy with stopwords of unigram is {0:.3f}".format(mnb_score_calculator(unigram)))
print("Accuracy with stopwords of bigram is {0:.3f}".format(mnb_score_calculator(bigram)))
print("Accuracy with stopwords of unigram_bigram is {0:.3f}".format(mnb_score_calculator(unigram_bigram)))

# Six category classifier
print("Accuracy with stopwords of six-category unigram is {0:.3f}".format(
    mnb_score_calculator(unigram, six_cat=True)))
print("Accuracy with stopwords of six-category bigram is {0:.3f}".format(
    mnb_score_calculator(bigram, six_cat=True)))
print("Accuracy with stopwords of six-category unigram_bigram is {0:.3f}".format(
    mnb_score_calculator(unigram_bigram, six_cat=True)))

module_analysis()
