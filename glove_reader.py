import numpy as np
import pickle
import os

filename_wordEmbedding = './glove/glove.6B.100d.txt'
filename_wordEmbedding_pickel = './PickledData/glove.pkl'

def make_glove_pickle():
    embeddings_index = {}

    # Open glove text file
    with open(filename_wordEmbedding, encoding='utf-8') as wordEmbedding_file:
        for line in wordEmbedding_file:
            values = line.split()
            try:
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
            except:
                pass

            # Write word vectors to dictionary
            embeddings_index[word] = vector

    if not os.path.exists('PickledData/'):
        print('MAKING DIRECTORY PickledData/ to save pickled glove file')
        os.makedirs('PickledData/')

    # Write glove pickle file
    with open(filename_wordEmbedding_pickel, 'wb') as glove_pickle:
        pickle.dump(embeddings_index, glove_pickle)

    print("- done. {} tokens".format(len(embeddings_index)))
make_glove_pickle()
