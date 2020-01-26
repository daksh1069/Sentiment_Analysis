#!/usr/bin/env python
# coding: utf-8


import numpy as np


from keras.datasets import imdb

top_words = 5000
(x_train,y_train), (x_test,y_test) = imdb.load_data(num_words=top_words)

print(x_train.shape) # This is a vector and not a matrix



print(x_train[0])


dict_w2c = imdb.get_word_index()
# print(dict_rev)



# Dictionary Comprehension
dict_c2w = { c+3:w for (w,c) in dict_w2c.items() } # + 3 isliye because some indexes are reserved for some special characters



r1 = x_train[0]
review = [ dict_c2w.get(c) for c in r1 ]
print(review)


from keras.preprocessing import sequence

max_review_length = 500
X_train = sequence.pad_sequences(x_train, maxlen = max_review_length)
X_test  = sequence.pad_sequences(x_test, maxlen = max_review_length)


print(X_train.shape)

# If Padding Happens in the Starting then, experimentally it was obvserved Accuarcy is higher. 


# Now we add a embedding layer to convert a word into vector


from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding


embedding_vector_length = 32

model = Sequential()
model.add(Embedding(top_words,embedding_vector_length , input_length = max_review_length))

# You may also add a conv layer here and Maxpool Layer to prevent overfiiting and feature Extraction
model.add
model.add(LSTM(100))

model.add(Dense(1,  activation = 'sigmoid' ))
model.compile(loss='binary_crossentropy',
             optimizer = 'adam',
             metrics =['accuracy'] )
model.summary()


# In[25]:


model.fit(X_train, y_train, epochs =1 , batch_size= 64)
score = model.evaluate(X_test,y_test, verbose = 2)
print(score)

revPos = "the movie was excellent and just brilliant cast was great"
revNeg = "the movie was the worst and just boring story was slow"


rev1 = revPos.split()
rev2 = [ dict_w2c.get(w) + 3 for w in rev1]
rev3 = [0]*(500-len(rev2)-1 ) + [1] +rev2  # Manual Padding 
rev4 = np.asarray(rev3).reshape(1,-1)
model.predict(rev4)



