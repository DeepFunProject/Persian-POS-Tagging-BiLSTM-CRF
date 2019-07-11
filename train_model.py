# ==============================================================================
#                                Imports
# ==============================================================================
import numpy as np
import pickle, sys, os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from keras_contrib.layers import CRF
from sklearn.model_selection import train_test_split

# ==============================================================================
#                                DATA ADDRESS
# ==============================================================================
dataset_Address = './pickleData/data.pkl'
glove_Address = './pickleData/glove.pkl'
saved_model_Address = './Models/BiLSTM_CRF_model.h5'

# ==============================================================================
#                          MAKE FOLDER TO SAVE MODEL
# ==============================================================================
if not os.path.exists('Models/'):
  print('MAKING DIRECTORY Models/ to save model file')
  os.makedirs('Models/')
  
# ==============================================================================
#                            HYPER-PARAMETERS
# ============================================================================== 
MAX_SEQUENCE_LENGTH = 300 # calculate for this corpus
EMBEDDING_DIM = 300 # glove embeddings
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 100
dropout = 0.5
n_epochs = 15

# ==============================================================================
#                                POS MODEL
# ==============================================================================
def pos_model():
  with open(dataset_Address, 'rb') as Bi_jan_khan_corpus:
    X, Y, word2int, int2word, tag2int, int2tag = pickle.load(Bi_jan_khan_corpus)
  
  X = pad_sequences(X[:70000], maxlen=MAX_SEQUENCE_LENGTH)
  Y = pad_sequences(Y[:70000], maxlen=MAX_SEQUENCE_LENGTH)
  Y = to_categorical(Y, num_classes=len(tag2int) + 1)
  
  print('TOTAL TAGS:', len(tag2int))
  print('TOTAL WORDS:', len(word2int))
  
  # Total number of tags
  n_tags = len(tag2int)
  
  indices = np.arange(X.shape[0])
  np.random.shuffle(indices)
  X = X[indices]
  Y = Y[indices]
  
  
  # split data into train and test
  X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                      test_size=TEST_SPLIT,
                                                      random_state=42)
    
  # split training data into train and validation
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    test_size=VALIDATION_SPLIT,
                                                    random_state=1)

  n_train_samples = X_train.shape[0]
  n_val_samples = X_val.shape[0]
  n_test_samples = X_test.shape[0]
  
  print('We have %d TRAINING samples' % n_train_samples)
  print('We have %d VALIDATION samples' % n_val_samples)
  print('We have %d TEST samples' % n_test_samples)
  print()

  with open(glove_Address, 'rb') as glove_file:
    embeddings_index = pickle.load(glove_file)
  
  print('Total %s word vectors.' % len(embeddings_index))
  print()
  
  embedding_matrix = np.random.random((len(word2int) + 1, EMBEDDING_DIM))
  
  for word, i in word2int.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector
  
  print('Embedding matrix shape', embedding_matrix.shape)
  print('x_train shape', X_train.shape)
  print('y_train shape', y_train.shape)
  print()
  
  inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  
  embedding_layer = Embedding(input_dim=len(word2int) + 1,
                              output_dim=EMBEDDING_DIM,
                              weights=[embedding_matrix],
                              input_length=MAX_SEQUENCE_LENGTH,
                              trainable=True)
  
  embedded_sequences = embedding_layer(inputs)
  
  lstm_0 = Bidirectional(LSTM(units=128, return_sequences=True, 
                         recurrent_dropout=0.2, activation='tanh'))(embedded_sequences)
  dense_0 = TimeDistributed(Dense(units=50, activation='relu'))(lstm_0)
  crf = CRF(n_tags+1)
  preds = crf(dense)
  
  model = keras.Model(inputs=inputs, outputs=preds)
  
  early = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.003,
                              patience=5,
                              verbose=1, mode='auto')

  model.compile(loss=crf.loss_function,
                optimizer=keras.optimizers.Adam(lr=0.001),
                metrics=[crf.accuracy])
  print("model fitting - Bidirectional LSTM_CRF")
  
  model.summary()
  
  history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=n_epochs, batch_size=BATCH_SIZE, callbacks=[early,], verbose=1)
  
  ploting_train_accuracy_name='crf_viterbi_accuracy'
  ploting_val_accuracy_name='val_crf_viterbi_accuracy'
  
  hist = pd.DataFrame(history.history)

  plt.style.use("ggplot")
  fig = plt.figure(figsize=(12,12))
  plt.plot(hist[ploting_train_accuracy_name], label="train_acc")
  plt.plot(hist[ploting_val_accuracy_name], label="val_acc")
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
             fancybox=True, shadow=True)
  plt.savefig('BiLstm_accuracy')
  plt.show()
    
  plt.style.use("ggplot")
  fig = plt.figure(figsize=(12,12))
  plt.plot(hist["loss"], label="train_loss")
  plt.plot(hist["val_loss"], label="val_loss")
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
             fancybox=True, shadow=True)
  plt.savefig('BiLstm_loss')
  plt.show()
  
  save_load_utils.save_all_weights(model, saved_model_Address)
  test_results = model.evaluate(X_test, y_test, verbose=1)
  print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))
  
  
pos_model()