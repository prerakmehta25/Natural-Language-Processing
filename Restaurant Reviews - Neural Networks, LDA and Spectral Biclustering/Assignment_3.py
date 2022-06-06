#!/usr/bin/env python
# coding: utf-8

# In[16]:


import multiprocessing

import re,string
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer,    CountVectorizer, HashingVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import DistanceMetric

from sklearn.manifold import TSNE

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import PorterStemmer
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[17]:


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# In[18]:


dataset.insert(0, 'id', range(0, 0 + len(dataset)))
dataset


# In[19]:


dataset.rename(columns={"Review": "body", "Liked": "Target"})

dataset.columns = ['id','body', 'target']
dataset.to_json('RestaurantReview.json')


# In[20]:


stoplist = nltk.corpus.stopwords.words('english')
DROP_STOPWORDS = False

#Functionality to turn stemming on or off
STEMMING = False  # judgment call, parsed documents more readable if False

MAX_NGRAM_LENGTH = 1  # try 1 for unigrams... 2 for bigrams... and so on
VECTOR_LENGTH = 1000  # set vector length for TF-IDF and Doc2Vec
WRITE_VECTORS_TO_FILE = False
SET_RANDOM = 9999


# In[21]:


def parse_words(text): 
    #lowe case all words
    text = text.lower()
    # split document into individual words
    tokens=text.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out tokens that are one or two characters long
    tokens = [word for word in tokens if len(word) > 2]
    # filter out tokens that are more than twenty characters long
    tokens = [word for word in tokens if len(word) < 21]
    # filter out stop words if requested
    if DROP_STOPWORDS:
        tokens = [w for w in tokens if not w in stoplist]         
    # perform word stemming if requested
    if STEMMING:
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
    # recreate the document string from parsed words
    text = ''
    for token in tokens:
        text = text + ' ' + token
    return tokens, text 


# In[22]:


dataset


# In[23]:


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))
    fig = plt.figure(1, figsize = (15, 15))
    plt.axis('off')
    if title: 
            fig.suptitle(title, fontsize = 20)
            fig.subplots_adjust(top = 2.3)
    plt.imshow(wordcloud)
    plt.show()

# print wordcloud
show_wordcloud(dataset['body'])


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.body, dataset.target, test_size = 0.20, random_state = 0)


# In[57]:


train_tokens = []  # list of token lists for gensim Doc2Vec
train_text = [] # list of document strings for sklearn TF-IDF
index = 0
label = []
for doc in X_train:
    text_string = doc
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    train_tokens.append(tokens)
    train_text.append(text_string)
    index = index + 1
    label.append(index)
print('\nNumber of training documents:',len(train_text))
print('\nFirst item after text preprocessing, train_text[0]\n', train_text[0])
print('\nNumber of training token lists:',len(train_tokens))
print('\nFirst list of tokens after text preprocessing, train_tokens[0]\n', train_tokens[0])
train_target = y_train 


# In[26]:


test_tokens = []  # list of token lists for gensim Doc2Vec
test_text = [] # list of document strings for sklearn TF-IDF
index = 0
label_test = []

for doc in X_test:
    text_string = doc
    # parse words one at a time in document string
    tokens, text_string = parse_words(text_string)
    test_tokens.append(tokens)
    test_text.append(text_string)
    index = index + 1
    label_test.append(index)
print('\nNumber of testing documents:',len(test_text))
print('\nFirst item after text preprocessing, test_text[0]\n', test_text[0])
print('\nNumber of testing token lists:',len(test_tokens))
print('\nFirst list of tokens after text preprocessing, test_tokens[0]\n', test_tokens[0])
test_target = y_test


# In[27]:


tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), max_features = VECTOR_LENGTH)

tfidf_vectors = tfidf_vectorizer.fit_transform(train_text)

print('\nTFIDF vectorization. . .')

print('\nTraining tfidf_vectors_training.shape:', tfidf_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use tfidf_vectorizer.transform, NOT tfidf_vectorizer.fit_transform

tfidf_vectors_test = tfidf_vectorizer.transform(test_text)

print('\nTest tfidf_vectors_test.shape:', tfidf_vectors_test.shape)

tfidf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = SET_RANDOM)

tfidf_clf.fit(tfidf_vectors, train_target)

tfidf_pred = tfidf_clf.predict(tfidf_vectors_test)  # evaluate on test set

print('\nTF-IDF/Random forest F1 classification performance in test set:', 
      round(metrics.f1_score(test_target, tfidf_pred, average='macro'), 3))


# In[28]:


matrix=pd.DataFrame(tfidf_vectors.toarray(), columns=tfidf_vectorizer.get_feature_names(), index=label)
matrix


# In[29]:


count_vectorizer = CountVectorizer(ngram_range = (1, MAX_NGRAM_LENGTH), max_features = VECTOR_LENGTH)

count_vectors = count_vectorizer.fit_transform(train_text)

print('\ncount vectorization. . .')

print('\nTraining count_vectors_training.shape:', count_vectors.shape)

# Apply the same vectorizer to the test data
# Notice how we use count_vectorizer.transform, NOT count_vectorizer.fit_transform
count_vectors_test = count_vectorizer.transform(test_text)

print('\nTest count_vectors_test.shape:', count_vectors_test.shape)

count_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = SET_RANDOM)

count_clf.fit(count_vectors, train_target)

count_pred = count_clf.predict(count_vectors_test)  # evaluate on test set

print('\nCount/Random forest F1 classification performance in test set:',
      round(metrics.f1_score(test_target, count_pred, average='macro'), 3))


# In[30]:


matrix1=pd.DataFrame(count_vectors.toarray(), columns=count_vectorizer.get_feature_names(), index=label)
matrix1


# ### As noted review number 798 in the training set has prevelance on word "you" from CountVectorizer and also has the has the highest TF-IDF  

# In[31]:


print('\nBegin Doc2Vec Work')
cores = multiprocessing.cpu_count()
print("\nNumber of processor cores:", cores)

train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_tokens)]
# print('train_corpus[:2]:', train_corpus[:2])

# Instantiate a Doc2Vec model with a vector size with 50 words 
# and iterating over the training corpus 40 times. 
# Set the minimum word count to 2 in order to discard words 
# with very few occurrences. 
# window (int, optional) – The maximum distance between the 
# current and predicted word within a sentence.
print("\nWorking on Doc2Vec vectorization, dimension 50")
model_50 = Doc2Vec(train_corpus, vector_size = 50, window = 4, min_count = 2, workers = cores, epochs = 40)

model_50.train(train_corpus, total_examples = model_50.corpus_count, epochs = model_50.epochs)  # build vectorization model on training set

# vectorization for the training set
#doc2vec_50_vectors = np.zeros((len(train_tokens), 50)) # initialize numpy array
doc2vec_50_df = pd.DataFrame()
for i in range(0, len(train_tokens)):
    doc2vec_50_vectors = pd.DataFrame(model_50.infer_vector(train_tokens[i])).transpose()
    doc2vec_50_df = pd.concat([doc2vec_50_df,doc2vec_50_vectors],axis=0)
    #doc2vec_50_vectors[i,] = model_50.infer_vector(train_tokens[i]).transpose()
print('\nTraining doc2vec_50_df.shape:', doc2vec_50_df.shape)
# print('doc2vec_50_vectors[:2]:', doc2vec_50_vectors[:2])

# vectorization for the test set

#doc2vec_50_vectors_test = np.zeros((len(test_tokens), 50)) # initialize numpy array
doc2vec_50_test_df = pd.DataFrame()
for i in range(0, len(test_tokens)):
    doc2vec_50_vectors_test = pd.DataFrame(model_50.infer_vector(test_tokens[i])).transpose()
    doc2vec_50_test_df = pd.concat([doc2vec_50_test_df,doc2vec_50_vectors_test],axis=0)
    #doc2vec_50_vectors_test[i,] = model_50.infer_vector(test_tokens[i]).transpose()
print('\nTest doc2vec_50_test_df.shape:', doc2vec_50_test_df.shape)

doc2vec_50_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = SET_RANDOM)
doc2vec_50_clf.fit(doc2vec_50_df, train_target) # fit model on training set
doc2vec_50_pred = doc2vec_50_clf.predict(doc2vec_50_test_df)  # evaluate on test set
print('\nDoc2Vec_50/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_50_pred, average='macro'), 3)) 


# In[32]:


doc2vec_50_df


# In[33]:


print("\nWorking on Doc2Vec vectorization, dimension 100")
model_100 = Doc2Vec(train_corpus, vector_size = 100, window = 4, min_count = 2, workers = cores, epochs = 40)

model_100.train(train_corpus, total_examples = model_100.corpus_count, epochs = model_100.epochs)  # build vectorization model on training set

doc2vec_100_df = pd.DataFrame()
for i in range(0, len(train_tokens)):
    doc2vec_100_vectors = pd.DataFrame(model_100.infer_vector(train_tokens[i])).transpose()
    doc2vec_100_df = pd.concat([doc2vec_100_df,doc2vec_100_vectors],axis=0)
print('\nTraining doc2vec_100_df.shape:', doc2vec_100_df.shape)

# vectorization for the test set

doc2vec_100_test_df = pd.DataFrame()
for i in range(0, len(test_tokens)):
    doc2vec_100_vectors_test = pd.DataFrame(model_100.infer_vector(test_tokens[i])).transpose()
    doc2vec_100_test_df = pd.concat([doc2vec_100_test_df,doc2vec_100_vectors_test],axis=0)
    #doc2vec_50_vectors_test[i,] = model_50.infer_vector(test_tokens[i]).transpose()
print('\nTest doc2vec_100_test_df.shape:', doc2vec_100_test_df.shape)

doc2vec_100_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = SET_RANDOM)
doc2vec_100_clf.fit(doc2vec_100_df, train_target) # fit model on training set
doc2vec_100_pred = doc2vec_100_clf.predict(doc2vec_100_test_df)  # evaluate on test set
print('\nDoc2Vec_100/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_100_pred, average='macro'), 3)) 


# In[34]:


doc2vec_100_df


# In[35]:


print("\nWorking on Doc2Vec vectorization, dimension 200")
model_200 = Doc2Vec(train_corpus, vector_size = 200, window = 4, min_count = 2, workers = cores, epochs = 40)

model_200.train(train_corpus, total_examples = model_200.corpus_count, epochs = model_200.epochs)  # build vectorization model on training set

doc2vec_200_df = pd.DataFrame()
for i in range(0, len(train_tokens)):
    doc2vec_200_vectors = pd.DataFrame(model_200.infer_vector(train_tokens[i])).transpose()
    doc2vec_200_df = pd.concat([doc2vec_200_df,doc2vec_200_vectors],axis=0)
print('\nTraining doc2vec_200_df.shape:', doc2vec_200_df.shape)

# vectorization for the test set

doc2vec_200_test_df = pd.DataFrame()
for i in range(0, len(test_tokens)):
    doc2vec_200_vectors_test = pd.DataFrame(model_200.infer_vector(test_tokens[i])).transpose()
    doc2vec_200_test_df = pd.concat([doc2vec_200_test_df,doc2vec_200_vectors_test],axis=0)
    #doc2vec_50_vectors_test[i,] = model_50.infer_vector(test_tokens[i]).transpose()
print('\nTest doc2vec_200_test_df.shape:', doc2vec_200_test_df.shape)

doc2vec_200_clf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = SET_RANDOM)
doc2vec_200_clf.fit(doc2vec_200_df, train_target) # fit model on training set
doc2vec_200_pred = doc2vec_200_clf.predict(doc2vec_200_test_df)  # evaluate on test set
print('\nDoc2Vec_200/Random forest F1 classification performance in test set:',
    round(metrics.f1_score(test_target, doc2vec_200_pred, average='macro'), 3)) 


# In[36]:


doc2vec_200_df


# In[37]:


TF_IDF_Performance = round(metrics.f1_score(test_target, tfidf_pred, average='macro'), 3)
count_vectorizer_Performance = round(metrics.f1_score(test_target, count_pred, average='macro'), 3)
Doc2Vec_50_Performance = round(metrics.f1_score(test_target, doc2vec_50_pred, average='macro'), 3)
Doc2Vec_100_Performance = round(metrics.f1_score(test_target, doc2vec_100_pred, average='macro'), 3)
Doc2Vec_200_Performance = round(metrics.f1_score(test_target, doc2vec_200_pred, average='macro'), 3)


# In[38]:


d = {'TF-IDF': [TF_IDF_Performance], 'count_vectorizer_Performance': [count_vectorizer_Performance], 
     'Doc2Vec_50_Performance': [Doc2Vec_50_Performance], 'Doc2Vec_100_Performance': [Doc2Vec_100_Performance], 
     'Doc2Vec_200_Performance': [Doc2Vec_200_Performance]}
df = pd.DataFrame(data=d)
df.index = ['F1_Score']
df
  


# In[40]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(matrix)
    Sum_of_squared_distances.append(km.inertia_)
    

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()   


# In[77]:


k=8
RANDOM_SEED = 9999

km = KMeans(n_clusters=k, random_state =RANDOM_SEED)
km.fit(matrix1)
clusters = km.labels_.tolist()

terms = tfidf_vectorizer.get_feature_names()
Dictionary={'Doc Id':label, 'Cluster':clusters,  'Text': train_text}
frame=pd.DataFrame(Dictionary, columns=['Cluster', 'Doc Id','Text'])

#frame=pd.concat([frame,data['category']], axis=1)

#frame['record']=1


# In[78]:


print("Top terms per cluster:")
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

terms_dict=[]

#save the terms for each cluster and document to dictionaries.  To be used later
#for plotting output.

# dictionary to store terms and titles
cluster_terms={}
cluster_title={}

for i in range(k):
    print("Cluster %d:" % i),
    temp_terms=[]
    temp_titles=[]
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        terms_dict.append(terms[ind])
        temp_terms.append(terms[ind])
    cluster_terms[i]=temp_terms
    
    print("Cluster %d titles:" % i, end='')
    temp=frame[frame['Cluster']==i]
    for title in temp['Doc Id']:
        print(' %s,' % title, end='')
        temp_titles.append(title)
    cluster_title[i]=temp_titles


# In[79]:


mds = TSNE(n_components=2, metric="euclidean", random_state=RANDOM_SEED)

pos = mds.fit_transform(count_vectors.toarray())

xs, ys = pos[:,0], pos[:,1]

cluster_colors = {0: 'black', 1: 'grey', 2: 'blue', 3: 'rosybrown', 4: 'firebrick', 
                  5: 'red', 6: 'darksalmon', 7: 'sienna'}

cluster_labels = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3',
                 4: 'Cluster 4', 5: 'Cluster 5', 6: 'Cluster 6', 7: 'Cluster 7'}

cluster_dict = cluster_title

df = pd.DataFrame(dict(x=xs, y= ys, label = clusters, title = range(0, len(clusters))))

groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12,12))
ax.margins(0.05)

for name, group in groups:
    ax.plot(group.x, group.y, marker ='o', linestyle ='', ms=12,
           label = cluster_labels[name], color = cluster_colors[name],
           mec = 'none')
    ax.set_aspect('auto')
    ax.tick_params(                   axis = 'x',
                   which = 'both',
                   bottom = 'off',
                   top = 'off',
                   labelbottom = 'on')
    ax.tick_params(                  axis = 'y',
                  which = 'both',
                  left = 'off',
                  top = 'off',
                  labelleft = 'on')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[44]:


type(tfidf_vectors)


# In[140]:


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Dendograms")
dend = shc.dendrogram(shc.linkage(matrix1, method='ward'))


# In[122]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
#y_hc = cluster.fit_predict(matrix1)
y_hc = cluster.fit_predict(count_vectors.toarray())

y_hc


# In[127]:


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
X, _ = make_multilabel_classification(random_state=0)
lda = LatentDirichletAllocation(n_components=5, random_state=0)

lda.fit(count_vectors.toarray())

lda_ft = lda.transform(count_vectors.toarray())

lda_ft


# In[139]:


from sklearn.cluster import SpectralBiclustering
import numpy as np

clustering = SpectralBiclustering(n_clusters=2, random_state=0).fit(count_vectors.toarray())
print(clustering.row_labels_)

print(clustering.column_labels_)

print(clustering)
    

