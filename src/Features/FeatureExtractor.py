from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from gensim import corpora, models
from gensim.corpora import Dictionary


import Operations.Operations as ops

class BagofWordFeatureExtractor:

    def __init__(self):
        '''
        creates a stemmer-object to get the stem from a given word
        stop-words set to look-up
        '''
        self.stemmer = SnowballStemmer("english")
        self.stopWords = set(stopwords.words("english"))

    def get_tokens(self, sentence):
        '''
        returns the tokens in the sentence
        "I like pizza" => [i, like, pizza]
        :param sentence:
        :return:
        '''
        tokens= wordpunct_tokenize(sentence)
        # manually remove punctuations
        for value in tokens:
            if value == "," or value == ";" or value==".":
                tokens.remove(value)
        return tokens

    def remove_stop_words(self, tokens):
        '''
        remove the list of stop words in nltk
        :param tokens:
        :return:
        '''
        for token in tokens:
            if token in self.stopWords:
                tokens.remove(token)
        return tokens

    def get_stems(self, tokens):
        '''
        for every token in the list, return the stem of the value
        :param tokens:
        :return:
        '''
        stems =[]
        for token in tokens:
            try:
                stems.append(self.stemmer.stem(token))
            except:
                #if random-exception hits append the existing word
                stems.append(token)
        return stems


    def get_bag_of_words(self, sentence):
        '''
        gets the  tokens the sentence
        removes the stop words from sentence
        returns the stem of every word in the sentence
        :return:
        '''
        tokens = self.get_tokens(sentence)
        filtered_tokens = self.remove_stop_words(tokens)
        stemmed_words = self.get_stems(filtered_tokens)
        return stemmed_words

    def get_corpus(self, sentences):
        sentence_bags = ops.parallel_execute(self.get_bag_of_words, sentences)
        texts_dict = corpora.Dictionary(sentence_bags)
        corpus = [texts_dict.doc2bow(text) for text in sentence_bags]
        return corpus



#testDataLoader = BagofWordFeatureExtractor()
#print(testDataLoader.get_bag_of_words("I want a 5$ pizza.Now I also want a 3$ pizza"))