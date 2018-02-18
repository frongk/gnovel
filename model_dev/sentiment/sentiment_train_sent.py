import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Embedding 
from keras.callbacks import ModelCheckpoint 
import os  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve 
import matplotlib.pyplot as plt 

import pickle as pickle

# Hyperparameters

# output directory name:
output_dir = 'sentiment_save'

# training:
epochs = 8
batch_size = 64

# vector-space embedding: 
n_dim = 64
n_unique_words = 5000 
max_sent_length = 30 
pad_type = trunc_type = 'pre'

# neural network architecture: 
n_dense = 64
dropout = 0.5

# Get Sample Data Ready
imdb = pd.read_csv('data/imdb_labelled.txt',sep='\t',header=None)
amazon = pd.read_csv('data/amazon_cells_labelled.txt',sep='\t',header=None)
yelp = pd.read_csv('data/yelp_labelled.txt',sep='\t',header=None)
data = pd.concat([imdb,amazon,yelp], axis=0)

token_tool = Tokenizer(
                      num_words=n_unique_words,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=" ",
                      char_level=False,
                     )

token_tool.fit_on_texts(data[0])

with open('model_output/tokenizer_obj.pkl', 'wb') as tokfile:
    pickle.dump(token_tool, tokfile)

X = token_tool.texts_to_sequences(data[0])
y = data[1]

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)


x_train = pad_sequences(x_train, maxlen=max_sent_length, padding=pad_type, truncating=trunc_type, value=0)
x_valid = pad_sequences(x_valid, maxlen=max_sent_length, padding=pad_type, truncating=trunc_type, value=0)

# Model Setup
model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_sent_length))
model.add(Flatten())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
modelcheckpoint = ModelCheckpoint(filepath=output_dir+"/weights.{epoch:02d}.hdf5")

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid), callbacks=[modelcheckpoint])

# Evaluate
model.load_weights(output_dir+"/weights.08.hdf5")
y_hat = model.predict_proba(x_valid)

pct_auc = roc_auc_score(y_valid, y_hat)*100.0
print( pct_auc)




