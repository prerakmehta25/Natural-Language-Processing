#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[156]:


import os
import numpy as np
import glob
from IPython.utils import io
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import save, load
from pickleshare import PickleShareDB
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[157]:


# positive reviews, text data (.txt)
posRev = 'pos'
# negative reviews, text data (.txt)
negRev = 'neg'
# Word embeddings
GloVeDir = 'gloVe.6B' 
# super-directory
movieDir = os.getcwd()


# In[158]:


# Pulling txt review files for each type of review. 
posFils=glob.glob(posRev+'/*.txt')
negFils=glob.glob(negRev+'/*.txt')
GloFils=glob.glob(GloVeDir+'/*.txt')


# In[159]:


# Just to check there are a total of 12,500 of each pos and neg reviews 
print(f'number of pos review files: {len(posFils)}')
print(f'number of neg review files: {len(negFils)}')
print(f'embedding files:\n {GloFils}')


# In[160]:


labels=[]      # labels list
reviews=[]     # reviews list, a list of text strings

for sentVal in ['pos','neg']:
    sentDir = os.path.join(movieDir,sentVal)
    print(sentDir)
    for filNam in glob.glob(sentDir+'/*.txt'):
        with open(filNam) as inFile:
            fil = inFile.read()
        reviews.append(fil)
        if sentVal == 'neg':
            labels.append(0)            # neg review -> label = 0
        else:
            labels.append(1)            # pos review -> label = 1


# In[161]:


Id = list(range(1, 25001))


# In[162]:


data = pd.DataFrame(list(zip(Id, labels,reviews)), 
               columns =['Id', 'labels','reviews']) 
 


# In[163]:


data.shape


# In[164]:


data.head()


# ### Data preprocessing

# In[165]:


import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
stemmer = PorterStemmer()

from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))


# In[166]:


def clean_text(text):
  clean_review=[]
  for i in text:
    
 
    text = re.sub(r'[^\w\s]','',i, re.UNICODE)
    text = text.lower()

    clean_review.append(text)
  return clean_review


# In[167]:


d1=data['reviews']
data_reviews=clean_text(d1)


# In[168]:


data['reviews']=pd.Series(data_reviews)
data.head()


# ### Tokenization
# #### Remove all words of len less than 3

# In[169]:


data['reviews'] = data['reviews'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


# In[170]:


tokenized_review = data['reviews'].apply(lambda x: x.split())
tokenized_review.head()


# #### Remove stop words

# In[171]:


def remove_stop(review):
  unstammed=[]
  for i in review:
    text = [word for word in i if not word in stop_words]
    text = " ".join(text)  
    unstammed.append(text)
  return unstammed


# In[172]:


unstammed=remove_stop(tokenized_review)

data['final_review']=pd.Series(unstammed)

data.head()


# #### Words in Negative sentiment review

# In[173]:


normal_words =' '.join([text for text in data['final_review'][data['labels'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# #### Words in Positive sentiment review

# In[174]:


normal_words =' '.join([text for text in data['final_review'][data['labels'] == 1]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[175]:


maxLen = 200     # maximum review length
trainDatNbr = 1000    # nufrmber of reviews that will be used for training
valDatNbr = 5000      # reviews for validation
testDatNbr = 5000     # data for final model evaluation
maxWords1 = 10000    # max number of words in case 1
maxWords2 = 30000    # max number of words in case 2
emDim1=50 # dimensions in case 1
emDim2=200 #dimensions in case 2


# In[176]:


# split into words, no more than max words parameter
tokenizer1 = Tokenizer(num_words=maxWords1)
tokenizer2 = Tokenizer(num_words=maxWords2)

tokenizer1.fit_on_texts(data['final_review'])
sequences1 = tokenizer1.texts_to_sequences(data['final_review'])

tokenizer2.fit_on_texts(data['final_review'])
sequences2 = tokenizer2.texts_to_sequences(data['final_review'])

paddedSeqs1 = pad_sequences(sequences1,maxLen) 
paddedSeqs2 = pad_sequences(sequences2,maxLen) 

labels = data['labels']   
labels = np.asarray(labels)


# In[177]:


wordIndx1=tokenizer1.word_index
print(len(wordIndx1))

wordIndx2=tokenizer2.word_index
print(len(wordIndx2))


# In[178]:


print(f'number of padded sequences 1: {len(paddedSeqs1)}')
print(f'number of padded sequences 2: {len(paddedSeqs2)}')


# In[179]:


# Random sort of labels and reviews

np.random.seed(55)

Indx1 = np.arange(len(paddedSeqs1))
Indx2 = np.arange(len(paddedSeqs2))

np.random.shuffle(Indx1)
np.random.shuffle(Indx2)

labels1=labels[Indx1]
labels2=labels[Indx2]

paddedSeqs1=paddedSeqs1[Indx1]
paddedSeqs2=paddedSeqs2[Indx2]


# In[180]:


# Splitting data

XTrain1 = paddedSeqs1[:trainDatNbr]   # training data
XTrain2 = paddedSeqs2[:trainDatNbr]   # training data

yTrain1 = labels1[:trainDatNbr]       # training labels
yTrain2 = labels2[:trainDatNbr]       # training labels

XVal1 = paddedSeqs1[trainDatNbr:trainDatNbr+valDatNbr]
XVal2 = paddedSeqs2[trainDatNbr:trainDatNbr+valDatNbr]

yVal1 = labels1[trainDatNbr:trainDatNbr+valDatNbr]
yVal2 = labels2[trainDatNbr:trainDatNbr+valDatNbr]

XTest1 = paddedSeqs1[trainDatNbr+valDatNbr:trainDatNbr+valDatNbr+testDatNbr]
XTest2 = paddedSeqs2[trainDatNbr+valDatNbr:trainDatNbr+valDatNbr+testDatNbr]

yTest1 = labels1[trainDatNbr+valDatNbr:trainDatNbr+valDatNbr+testDatNbr]
yTest2 = labels2[trainDatNbr+valDatNbr:trainDatNbr+valDatNbr+testDatNbr]


# In[181]:


# checks on shapes
XTrain1.shape
yTrain1.shape
XVal1.shape
yVal1.shape
XTest1.shape
yTest1.shape

XTrain2.shape
yTrain2.shape
XVal2.shape
yVal2.shape
XTest2.shape
yTest2.shape


# In[182]:


embeddings_directory = 'gloVe.6B'
filename1 = 'glove.6B.50d.txt'
GloVeFil1 = os.path.join(embeddings_directory, filename1)
filename2 = 'glove.6B.200d.txt'
GloVeFil2 = os.path.join(embeddings_directory, filename2)

emIndx1=dict()
emIndx2=dict()

with open(GloVeFil1) as inFile:
    emFil=inFile.readlines()
cnt = 0
for line in emFil:      # reading line by line
    vals = line.split()
    word = vals[0]   # word is first value in each line
    coefs = np.asarray(vals[1:],dtype='float32')  # rest of line read into np array
    emIndx1[word]=coefs

with open(GloVeFil2) as inFile:
    emFil=inFile.readlines()
cnt = 0
for line in emFil:      # reading line by line
    vals = line.split()
    word = vals[0]   # word is first value in each line
    coefs = np.asarray(vals[1:],dtype='float32')  # rest of line read into np array
    emIndx2[word]=coefs


# In[183]:


print(f'number of vectors in 1 {len(emIndx1)}')
print(f'number of vectors in 2 {len(emIndx2)}')


# In[184]:


emMat1=np.zeros((maxWords1,emDim1))   # start with all zeros
emMat2=np.zeros((maxWords1,emDim2))   
emMat3=np.zeros((maxWords2,emDim1))  
emMat4=np.zeros((maxWords2,emDim2))   

for word, i in wordIndx1.items():
    if i < maxWords1:
        emVec = emIndx1.get(word)  
        if emVec is not None:
            emMat1[i]=emVec

for word, i in wordIndx1.items():
    if i < maxWords1:
        emVec = emIndx2.get(word)  
        if emVec is not None:
            emMat2[i]=emVec

for word, i in wordIndx2.items():
    if i < maxWords2:
        emVec = emIndx1.get(word)  
        if emVec is not None:
            emMat3[i]=emVec

for word, i in wordIndx2.items():
    if i < maxWords2:
        emVec = emIndx2.get(word)  # default is None when word is not in the index
        if emVec is not None:
            emMat4[i]=emVec


# In[185]:


print(emMat1.shape)
print(emMat2.shape)
print(emMat3.shape)
print(emMat4.shape)


# In[186]:



model1 = Sequential()
model1.add(Embedding(maxWords1,emDim1,input_length=maxLen))
model1.add(Bidirectional(LSTM(32, return_sequences = True)))
model1.add(GlobalMaxPool1D())
model1.add(Dense(20, activation="relu"))
model1.add(Dropout(0.05))
model1.add(Dense(1, activation="sigmoid"))

model2 = Sequential()
model2.add(Embedding(maxWords1,emDim2,input_length=maxLen))
model2.add(Bidirectional(LSTM(32, return_sequences = True)))
model2.add(GlobalMaxPool1D())
model2.add(Dense(20, activation="relu"))
model2.add(Dropout(0.05))
model2.add(Dense(1, activation="sigmoid"))

model3 = Sequential()
model3.add(Embedding(maxWords2,emDim1,input_length=maxLen))
model3.add(Bidirectional(LSTM(32, return_sequences = True)))
model3.add(GlobalMaxPool1D())
model3.add(Dense(20, activation="relu"))
model3.add(Dropout(0.05))
model3.add(Dense(1, activation="sigmoid"))

model4 = Sequential()
model4.add(Embedding(maxWords2,emDim2,input_length=maxLen))
model4.add(Bidirectional(LSTM(32, return_sequences = True)))
model4.add(GlobalMaxPool1D())
model4.add(Dense(20, activation="relu"))
model4.add(Dropout(0.05))
model4.add(Dense(1, activation="sigmoid"))

model5 = Sequential()
model5.add(Embedding(maxWords2, emDim2, input_length = maxLen))
model5.add(Convolution1D(filters = 32, kernel_size = 8, activation = 'relu'))
model5.add(GlobalMaxPool1D())
model5.add(Dense(20, activation = 'relu'))
model5.add(Dropout(0.05))
model5.add(Dense(1, activation = 'softmax'))


# In[187]:


model1.layers[0].set_weights([emMat1])  
model1.layers[0].trainable=False

model2.layers[0].set_weights([emMat2])  
model2.layers[0].trainable=False

model3.layers[0].set_weights([emMat3])  
model3.layers[0].trainable=False

model4.layers[0].set_weights([emMat4])  
model4.layers[0].trainable=False

model5.layers[0].set_weights([emMat4])  
model5.layers[0].trainable=False


# In[188]:


model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# In[189]:


import time

start_time = time.process_time()

history1=model1.fit(XTrain1, yTrain1,
        epochs=10,
        batch_size=100,
        validation_data=(XVal1,yVal1),
        verbose=0
        )

end_time = time.process_time()
runtime1 = end_time - start_time
print("\nProcessing time For Model 1 (seconds): %f" % runtime1)

start_time = time.process_time()
history2=model2.fit(XTrain1, yTrain1,
        epochs=10,
        batch_size=100,
        validation_data=(XVal1,yVal1),
        verbose=0
        )

end_time = time.process_time()
runtime2 = end_time - start_time
print("\nProcessing time For Model 2 (seconds): %f" % runtime2)

start_time = time.process_time()
history3=model3.fit(XTrain2, yTrain2,
        epochs=10,
        batch_size=100,
        validation_data=(XVal2,yVal2),
        verbose=0
        )

end_time = time.process_time()
runtime3 = end_time - start_time
print("\nProcessing time For Model 3 (seconds): %f" % runtime3)

start_time = time.process_time()
history4=model4.fit(XTrain2, yTrain2,
        epochs=10,
        batch_size=100,
        validation_data=(XVal2,yVal2),
        verbose=0
        )

end_time = time.process_time()
runtime4 = end_time - start_time
print("\nProcessing time For Model 4 (seconds): %f" % runtime4)


# In[190]:


earlystop_callback =     tf.keras.callbacks.EarlyStopping(monitor='val_acc',    min_delta=0.01, patience=5, verbose=0, mode='auto',    baseline=None, restore_best_weights=False)

start_time = time.process_time()
history5=model5.fit(XTrain2, yTrain2,
        epochs=100,
        batch_size=100,
        validation_data=(XVal2,yVal2),
        verbose=0
        ,callbacks = [earlystop_callback]
        )

end_time = time.process_time()
runtime5 = end_time - start_time
print("\nProcessing time For Model 5 (seconds): %f" % runtime5)


# In[191]:


Model1_accResults = pd.DataFrame(history1.history["acc"])
Model1_vaccResults = pd.DataFrame(history1.history["val_acc"])

Model2_accResults = pd.DataFrame(history2.history["acc"])
Model2_vaccResults = pd.DataFrame(history2.history["val_acc"])

Model3_accResults = pd.DataFrame(history3.history["acc"])
Model3_vaccResults = pd.DataFrame(history3.history["val_acc"])

Model4_accResults = pd.DataFrame(history4.history["acc"])
Model4_vaccResults = pd.DataFrame(history4.history["val_acc"])

epochs = 10

accResults = pd.concat([Model1_accResults, Model2_accResults, 
                        Model3_accResults, Model4_accResults], axis=1)
accResults.columns =["Model 1 Training Accuracy", "Model 2 Training Accuracy", 
                     "Model 3 Training Accuracy", "Model 4 Training Accuracy"] 
accResults.index =(np.arange(1, ((epochs)+1), step=1)) #this resets the index to epoch number

vaccResults = pd.concat([Model1_vaccResults, Model2_vaccResults, 
                         Model3_vaccResults, Model4_vaccResults], axis=1)
vaccResults.columns =["Model 1 Validation Accuracy", "Model 2 Validation Accuracy", 
                      "Model 3 Validation Accuracy", "Model 4 Validation Accuracy"] 
vaccResults.index =(np.arange(1, ((epochs)+1), step=1))

four_colors = ["Blue", "Green", "Red", "Black"]

fig = plt.figure(figsize=(21,6));  
ax1 = accResults.plot(kind='line', lw=2, style=':', color=(four_colors)); 
vaccResults.plot(kind='line', lw=3, color=(four_colors), ax=ax1); 
plt.title("Training and Validation Accuracy\nComparison of 4 Models\n", size=18);
plt.legend(loc='center right', bbox_to_anchor=(1.42, 0.5)); 
plt.xticks(np.arange(1, ((epochs)+1), step=1));
plt.yticks(np.arange(0.50, 1.01, step=0.03)); 
plt.xlabel('Epochs'); 
plt.ylabel('Accuracy');
plt.show()


# In[192]:


Model5_accResults = pd.DataFrame(history5.history["acc"])
Model5_vaccResults = pd.DataFrame(history5.history["val_acc"])


# In[193]:


from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

prediction1 = model1.predict(XTest1)
prediction2 = model2.predict(XTest1)
prediction3 = model3.predict(XTest2)
prediction4 = model4.predict(XTest2)
prediction5 = model5.predict(XTest2)

prediction1_train = model1.predict(XTrain1)
prediction2_train = model2.predict(XTrain1)
prediction3_train = model3.predict(XTrain2)
prediction4_train = model4.predict(XTrain2)
prediction5_train = model5.predict(XTrain2)

prediction1_val = model1.predict(XVal1)
prediction2_val = model2.predict(XVal1)
prediction3_val = model3.predict(XVal2)
prediction4_val = model4.predict(XVal2)
prediction5_val = model5.predict(XVal2)

y_pred1 = (prediction1 > 0.5)
y_pred2 = (prediction2 > 0.5)
y_pred3 = (prediction3 > 0.5)
y_pred4 = (prediction4 > 0.5)
y_pred5 = (prediction5 > 0.5)

y_pred1_train = (prediction1_train > 0.5)
y_pred2_train = (prediction2_train > 0.5)
y_pred3_train = (prediction3_train > 0.5)
y_pred4_train = (prediction4_train > 0.5)
y_pred5_train = (prediction5_train > 0.5)

y_pred1_val = (prediction1_val > 0.5)
y_pred2_val = (prediction2_val > 0.5)
y_pred3_val = (prediction3_val > 0.5)
y_pred4_val = (prediction4_val > 0.5)
y_pred5_val = (prediction5_val > 0.5)

cm1 = confusion_matrix(y_pred1, yTest1)
cm2 = confusion_matrix(y_pred2, yTest1)
cm3 = confusion_matrix(y_pred3, yTest2)
cm4 = confusion_matrix(y_pred4, yTest2)
cm5 = confusion_matrix(y_pred5, yTest2)

accuracy_score1 = accuracy_score(yTest1, y_pred1)
accuracy_score2 = accuracy_score(yTest1, y_pred2)
accuracy_score3 = accuracy_score(yTest2, y_pred3)
accuracy_score4 = accuracy_score(yTest2, y_pred4)
accuracy_score5 = accuracy_score(yTest2, y_pred5)

accuracy_score1_train = accuracy_score(yTrain1, y_pred1_train)
accuracy_score2_train = accuracy_score(yTrain1, y_pred2_train)
accuracy_score3_train = accuracy_score(yTrain2, y_pred3_train)
accuracy_score4_train = accuracy_score(yTrain2, y_pred4_train)
accuracy_score5_train = accuracy_score(yTrain2, y_pred5_train)

accuracy_score1_val = accuracy_score(yVal1, y_pred1_val)
accuracy_score2_val = accuracy_score(yVal1, y_pred2_val)
accuracy_score3_val = accuracy_score(yVal2, y_pred3_val)
accuracy_score4_val = accuracy_score(yVal2, y_pred4_val)
accuracy_score5_val = accuracy_score(yVal2, y_pred5_val)


# In[194]:


print('Confusion matrix of test results for model 1:\n', cm1)
print('Test Accuracy score for model 1: ', accuracy_score1)
print('Train Accuracy score for model 1: ', accuracy_score1_train)
print('Validation Accuracy score for model 1: ', accuracy_score1_val)
print('\n')

print('Confusion matrix of test results for model 2:\n', cm2)
print('Test Accuracy score for model 2: ', accuracy_score2)
print('Train Accuracy score for model 2: ', accuracy_score2_train)
print('Validation Accuracy score for model 2: ', accuracy_score2_val)
print('\n')

print('Confusion matrix of test results for model 3:\n', cm3)
print('Test Accuracy score for model 3: ', accuracy_score3)
print('Train Accuracy score for model 3: ', accuracy_score3_train)
print('Validation Accuracy score for model 3: ', accuracy_score3_val)
print('\n')

print('Confusion matrix of test results for model 4:\n', cm4)
print('Test Accuracy score for model 4: ', accuracy_score4)
print('Train Accuracy score for model 4: ', accuracy_score4_train)
print('Validation Accuracy score for model 4: ', accuracy_score4_val)

print('Confusion matrix of test results for model 5:\n', cm5)
print('Test Accuracy score for model 5: ', accuracy_score5)
print('Train Accuracy score for model 5: ', accuracy_score5_train)
print('Validation Accuracy score for model 5: ', accuracy_score5_val)


# In[195]:


data = {'Model Number': ['1','2','3','4','5'],
        'Vocabulary Size': ['10000','10000','30000','30000','30000'],
        'Pre-trained Word Embedding': ['GloVe.6B.50d','GloVe.6B.200d','GloVe.6B.50d','GloVe.6B.200d','Conv1D'],
        'Processing Time (seconds)': [runtime1,runtime2,runtime3,runtime4,runtime5],
        'Training Set Accuracy': [accuracy_score1_train,accuracy_score2_train,
                                  accuracy_score3_train,accuracy_score4_train,
                                  accuracy_score5_train],
        'Validation Set Accuracy': [accuracy_score1_val,accuracy_score2_val,
                                  accuracy_score3_val,accuracy_score4_val,
                                  accuracy_score5_val],
        'Test Set Accuracy': [accuracy_score1,accuracy_score2,accuracy_score3,accuracy_score4,accuracy_score5]}

df= pd.DataFrame(data)

df

