from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
from Features.FeatureExtractor import BagofWordFeatureExtractor
from gensim import corpora, models
import gensim
import nltk
import pandas as pd
import gzip
import pickle
import unicodedata
from Features.similarityCompute import *
import sys

model = pickle.load(open("finalized_model.sav", "rb"))
fast_text_model = pickle.load(open('fast_text_model.sav', 'rb'))
texts_dict = corpora.Dictionary.load("auto_review.dict")
raw_query = sys.argv[1]

query_words = raw_query.split()
tokenizer = RegexpTokenizer(r'\w+')

nltk_stpwd = stopwords.words('english')
stop_words_stpwd = get_stop_words('en')
merged_stopwords = list(set(nltk_stpwd + stop_words_stpwd))
sb_stemmer = SnowballStemmer("english")
bg = BagofWordFeatureExtractor()
query = bg.get_bag_of_words(raw_query)

id2word = gensim.corpora.Dictionary()
id2word.merge_with(texts_dict)
query = id2word.doc2bow(query)

a = list(sorted(model[query], key=lambda x: x[1]))
# print(a)
#print model.show_topics(num_topics=10, num_words=5)
topics = model.show_topics(num_topics=10, num_words=5)
labels = []

for topic in topics:
    values = topic[1].split("+")
    classlabel = []
    for value in values:
        score, word = value.split("*")
        word = word.replace("\"", "")
        classlabel.append(word)
    labels.append(classlabel)

# print "**labels**"
# label_vec = []
# for label in labels:
#     labelvec = []
#     for word in label:
#         labelvec.append(fast_text_model[word])
#     label_vec.append(labelvec)


op = model.print_topic(a[-1][0])
op = op.decode('utf-8')
print "********* The Output Department is ***********"


values = op.split("+")
scores = []
words = []
for value in values:
    score, word = value.split("*")
    word = word.replace("\"", "")
    scores.append(score)
    words.append(word)

print u", ".join(words)
result = []

for word in words:
    result.append(fast_text_model[word])
max_similarity = 0
best_label = 0

label_vec = pickle.load(open('label_vec.dict', 'rb'))
for i, lv in enumerate(label_vec):
    cosineSimilarity = similarity_compute(result, lv)
    #print cosineSimilarity
    if(max_similarity < cosineSimilarity):
        max_similarity = cosineSimilarity
        best_label = labels[i]

#print best_label
