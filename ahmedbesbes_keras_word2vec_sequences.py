import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import os
import csv
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
nltlTokenizer = TweetTokenizer()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import scale

from keras.models import Sequential
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input,  Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Embedding
from keras.callbacks import ModelCheckpoint

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove/'
GLOVE_FILE = 'glove.twitter.27B.200d.txt'
MAX_SEQ_LEN = 256
n_dim=200
bBuildWordVector = 0

#define a function that loads the dataset and extracts the two columns
def ingest():
    data = pd.read_csv('./training.1600000.processed.noemoticon.csv', encoding = "utf-8")
    data.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
    data = data[data.Sentiment.isnull() == False]
    data['Sentiment'] = data['Sentiment'].map( {4:1, 0:0} )
    #data['Sentiment'] = data['Sentiment'].map(int)
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print('dataset loaded with shape', data.shape)
    return data

data = ingest()
print(data.head(5))
#n=data.shape[0]
n=256000

def filterTweet(tweet):
    try:
        #tweet = unicode(tweet.decode('utf-8').lower())
        tweet = tweet.lower()
        tokens = nltlTokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return ' '.join(list(tokens))
    except:
        return 'NC'

#tokenizing function that splits each tweet into tokens and removes user mentions, hashtags and urls
def tokenize(tweet):
    try:
        #tweet = unicode(tweet.decode('utf-8').lower())
        tweet = tweet.lower()
        tokens = nltlTokenizer.tokenize(tweet)
        return list(tokens)
    except:
        return 'NC'

#The results of the tokenization should now be cleaned to remove lines with 'NC', resulting from a tokenization error
def postprocess(data, n=1600000):
    data = data.head(n)
    data['SentimentTextFiltered'] = data['SentimentText'].progress_map(filterTweet)
    data['tokens'] = data['SentimentTextFiltered'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.SentimentTextFiltered != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data['SentimentTextFiltered'], data['Sentiment'], data['tokens'].as_matrix()

X_raw, Y_raw, X_tokens = postprocess(data,n)
tokenizer = Tokenizer() #nb_words=MAX_NB_WORDS
tokenizer.fit_on_texts(X_raw)
sequences = tokenizer.texts_to_sequences(X_raw)
word_index = tokenizer.word_index
X_processed = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)
Y_processed = to_categorical(np.asarray(Y_raw), 2)

#Build the word2vec model
x_train, x_test, y_train, y_test = train_test_split(np.array(X_processed),
                                                    np.array(Y_processed), test_size=0.2)

# def labelizeTweets(tweets, label_type):
    # labelized = []
    # for i,v in tqdm(enumerate(tweets)):
        # label = '%s_%s'%(label_type,i)
        # labelized.append(LabeledSentence(v, [label]))
    # return labelized

# x_train = labelizeTweets(x_train, 'TRAIN')
# x_test = labelizeTweets(x_test, 'TEST')
print(x_train[0])

def get_embeddings():
    embeddings = {}
    with open(os.path.join(GLOVE_DIR, GLOVE_FILE), 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
    return embeddings

if(bBuildWordVector):
    tweet_w2v = Word2Vec(size=n_dim, min_count=10)
    tweet_w2v.build_vocab([x for x in tqdm(X_tokens)])
    tweet_w2v.train([x for x in tqdm(X_tokens)],total_examples=tweet_w2v.corpus_count,epochs=tweet_w2v.iter)
    #test built word2vec model
    if 'good' in tweet_w2v:
        print(tweet_w2v['good'])
        print(tweet_w2v.most_similar('good'))
else:    
    embeddings = get_embeddings()

# # importing bokeh library for interactive dataviz
# import bokeh.plotting as bp
# from bokeh.models import HoverTool, BoxSelectTool
# from bokeh.plotting import figure, show, output_notebook

# # defining the chart
# output_notebook()
# plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
    # tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    # x_axis_type=None, y_axis_type=None, min_border=1)

# # getting a list of word vectors. limit to 10000. each is of 200 dimensions
# word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:5000]]

# # dimensionality reduction. converting the vectors to 2d vectors
# from sklearn.manifold import TSNE
# tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
# tsne_w2v = tsne_model.fit_transform(word_vectors)

# # putting everything in a dataframe
# tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
# tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:5000]

# # plotting. the corresponding word appears when you hover on the data point.
# plot_tfidf.scatter(x='x', y='y', source=tsne_df)
# hover = plot_tfidf.select(dict(type=HoverTool))
# hover.tooltips={"word": "@words"}
# show(plot_tfidf)

# print('building tf-idf matrix ...')
# vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
# matrix = vectorizer.fit_transform([x.words for x in x_train])
# tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
# print('vocab size :', len(tfidf))

# def buildWordVector(tokens, size):
    # vec = np.zeros(size).reshape((1, size))
    # count = 0.
    # for word in tokens:
        # try:
            # vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            # count += 1.
        # except KeyError: # handling the case where the token is not
                         # # in the corpus. useful for testing.
            # continue
    # if count != 0:
        # vec /= count
    # return vec

# print('building train combines word_vectors with tf-idf ...')
# train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
# train_vecs_w2v = scale(train_vecs_w2v)
# print('train_vecs_w2v shape', train_vecs_w2v.shape)
# print('building test combines word_vectors with tf-idf ...')
# test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
# test_vecs_w2v = scale(test_vecs_w2v)
# print('test_vecs_w2v shape', test_vecs_w2v.shape)

def make_embedding_layer(word_index):    
    #nb_words = min(MAX_NB_WORDS, len(word_index))
    nb_words = len(word_index)
    embedding_matrix = np.zeros((nb_words, n_dim))

    for word, i in word_index.items():
        #if i >= MAX_NB_WORDS:
        #    continue
        if(bBuildWordVector):
            if word in tweet_w2v:
                embedding_vector = tweet_w2v[word].reshape((1, n_dim))
            else:
                embedding_vector = None
        else:
            embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(nb_words, n_dim, weights=[embedding_matrix], input_length=MAX_SEQ_LEN, trainable=False)
    return embedding_layer

print('begin to train DNN model for sentiment analysis...')
model = Sequential()
embedded_sequences = make_embedding_layer(word_index)
model.add(embedded_sequences)
model.add(Conv1D(256, 5, activation='relu'))
model.add(AveragePooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(labels_index), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=256, verbose=2)

print('Evaluate trained model on test dataset...')
score = model.evaluate(x_test, y_test, batch_size=256, verbose=2)
print('Accuracy: ', score[1])
