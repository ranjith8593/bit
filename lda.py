from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
from gensim.models import FastText
from gensim import corpora, models
import gensim
import nltk
import pandas as pd
import gzip
import pickle


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


df = getDF('reviews_Automotive_5.json.gz')

tokenizer = RegexpTokenizer(r'\w+')

doc_1 = df.reviewText[0]
tokens = tokenizer.tokenize(doc_1.lower())
nltk_stpwd = stopwords.words('english')
stop_words_stpwd = get_stop_words('en')
merged_stopwords = list(set(nltk_stpwd + stop_words_stpwd))
stopped_tokens = [token for token in tokens if not token in merged_stopwords]
sb_stemmer = SnowballStemmer('english')
stemmed_tokens = [sb_stemmer.stem(token) for token in stopped_tokens]

num_reviews = df.shape[0]

doc_set = [df.reviewText[i] for i in range(num_reviews)]

texts = []


for doc in doc_set:
    # putting our three steps together
    tokens = tokenizer.tokenize(doc.lower())
    stopped_tokens = [
        token for token in tokens if not token in merged_stopwords]
    stemmed_tokens = [sb_stemmer.stem(token) for token in stopped_tokens]

    # add tokens to list
    texts.append(stemmed_tokens)

texts_dict = corpora.Dictionary(texts)
texts_dict.save('auto_review_2.dict')
fast_text_model = FastText(texts, min_count=1)
pickle.dump(fast_text_model, open('fast_text_model.sav', 'wb'))
import operator
print("IDs 1 through 10: {}".format(sorted(texts_dict.token2id.items(),
                                           key=operator.itemgetter(1), reverse=False)[:10]))

corpus = [texts_dict.doc2bow(text) for text in texts]
# len(corpus)
# print corpus
gensim.corpora.MmCorpus.serialize('amzn_auto_review.mm', corpus)
#
lda_model = gensim.models.LdaModel(
    corpus, alpha='auto', num_topics=5, id2word=texts_dict, passes=20)
filename = 'finalized_model_2.sav'
pickle.dump(lda_model, open(filename, 'wb'))
topics = lda_model.show_topics(num_topics=10, num_words=5)
labels = []

for topic in topics:
    print topic
    values = topic[1].split("+")
    classlabel = []
    for value in values:
        score, word = value.split("*")
        word = word.replace("\"", "")
        classlabel.append(word)
    labels.append(classlabel)

print "**labels**"
print labels
label_vec = []
for label in labels:
    labelvec = []
    for word in label:
        labelvec.append(fast_text_model[word])
    label_vec.append(labelvec)
pickle.dump(labels, open('labels.dict', 'wb'))
pickle.dump(label_vec, open('label_vec.dict', 'wb'))
