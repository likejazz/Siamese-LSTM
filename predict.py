from time import time
import pandas as pd
import numpy as np
import sys

import itertools
import datetime

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import util
from util import make_w2v_embeddings
from util import ManDist

# File paths
TEST_CSV = './data/test-20.csv'
EMBEDDING_FILE = './data/GoogleNews-vectors-negative300.bin.gz'

# Load training set
test_df = pd.read_csv(TEST_CSV)
for q in ['question1', 'question2']:
    test_df[q + '_n'] = test_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
test_df, embeddings = make_w2v_embeddings(test_df, file=EMBEDDING_FILE, embedding_dim=embedding_dim)

# Split to dicts
X_test = {'left': test_df['question1_n'], 'right': test_df['question2_n']}

# Zero padding
for dataset, side in itertools.product([X_test], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

# --

model = load_model('./data/malstm.h5', custom_objects={'ManDist': ManDist})
model.summary()

prediction = model.predict([X_test['left'], X_test['right']])
print(prediction)
