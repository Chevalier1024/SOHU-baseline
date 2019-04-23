from keras.layers import *
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Dropout, Bidirectional
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from gensim.models import word2vec, KeyedVectors
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import jieba
import re
import tensorflow as tf
import json
from tqdm import tqdm

MAX_SEQUENCE_LENGTH = 600
EMBEDDING_DIM = 128
emotion2id = {'POS': 1, 'NORM': 0, 'NEG': 2}
id2emotion = {j:i for i,j in emotion2id.items()}

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def call(self, x):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.softmax(ait)
        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def processing_data(data, mode='train'):
    result = []
    label = []
    for line in data:
        text = line['content']
        if mode != 'test':
            l = emotion2id[line['emotion']]
            label.append(l)
        text = re.sub(r' ', '', text)
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[’!"#$%&\'()*+,-./:;<=>?@，。★、…【】《》？“”‘！^_`{|}~]+', '', text)
        words = list(jieba.cut(text))

        result.append(' '.join(words))
    return result, label


def get_model(embedding, class_num=3):
    inputs_sentence = Input(shape=(MAX_SEQUENCE_LENGTH,))

    sentence = SpatialDropout1D(0.2)(embedding(inputs_sentence))
    context = Bidirectional(CuDNNLSTM(128, return_sequences=True))(sentence) # 45 128

    max_pool = GlobalMaxPooling1D()(context)
    atten = AttentionWithContext()(context)
    x = concatenate([max_pool, atten], axis=1)
    x = Dropout(0.2)(x)
    output = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=[inputs_sentence], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def read_stopwords():
    stopwords = []
    with open('../../../data/stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                stopwords.append(line.strip())
    return stopwords


print('(1) read data and cut data')
train_data = json.load(open('./sentiment_data/train_data.json'))
dev_data = json.load(open('./sentiment_data/dev_data.json'))
test_data = json.load(open('./sentiment_data/test_data.json'))

x_train, y_train = processing_data(train_data)
x_dev, y_dev = processing_data(dev_data)
x_test, _ = processing_data(test_data, mode='test')

all_text = x_train + x_dev + x_test

print('(2) doc to var....')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)
train_sequence = tokenizer.texts_to_sequences(x_train)
dev_sequence = tokenizer.texts_to_sequences(x_dev)
test_sequence = tokenizer.texts_to_sequences(x_test)

y_train = to_categorical(y_train)
y_train = y_train.astype(np.int32)

y_dev_cat = to_categorical(y_dev)
y_dev_cat = y_dev_cat.astype(np.int32)


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
train_pad = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH)
dev_pad = pad_sequences(dev_sequence, maxlen=MAX_SEQUENCE_LENGTH)
test_pad = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)


print('(4) Preparing embedding matrix.')
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM), dtype=np.float32)
not_in_model = 0
in_model = 0
embedding_max_value = 0
embedding_min_value = 1
w2v_model = KeyedVectors.load_word2vec_format('./w2v/word2vec.txt')
not_words = []
for word, i in word_index.items():
    if word in w2v_model:
        in_model += 1
        embedding_matrix[i] = np.array(w2v_model[word])
        embedding_max_value = max(np.max(embedding_matrix[i]), embedding_max_value)
        embedding_min_value = min(np.min(embedding_matrix[i]), embedding_min_value)
    else:
        # embedding_matrix[i] = np.random.uniform(low=embedding_min_value, high=embedding_max_value, size=EMBEDDING_DIM)
        not_in_model += 1
        not_words.append(word)
print(str(not_in_model)+' words not in w2v model')



embed = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
model = get_model(embed)
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [EarlyStopping(monitor='val_acc', min_delta=0.001, verbose=1, patience=5),
             ModelCheckpoint('./sentiment_data/lstm_attention.check', monitor='val_acc',
                             mode='max', verbose=0, save_best_only=True, save_weights_only=True)]
model.fit(train_pad, y_train, batch_size=32, epochs=30, validation_data=(dev_pad, y_dev_cat), callbacks=callbacks)

model.load_weights(filepath='./sentiment_data/lstm_attention.check')
y_pedv = model.predict(dev_pad, batch_size=32)
y_pedv = np.argmax(y_pedv, axis=1).astype(int)
print(f1_score(y_dev, y_pedv, average='micro'))
print(classification_report(y_dev, y_pedv))


y_test = model.predict(test_pad, batch_size=32)
y_test = np.argmax(y_test, axis=1).astype(int)
np.savetxt('./sentiment_data/test_lstm_attention.txt', y_test)


# output result
id_entity_emotion = dict()

for i, line in enumerate(test_data):
    newsId = line['newsId']
    entity = line['entity']
    emotion = id2emotion[y_test[i]]
    id_entity_emotion[newsId+entity] = emotion


result = []
with open('./output/ner.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        if len(line) == 3:
            entity = line[1].split(',')
            emotion = []
            for e in entity:
                emotion.append(id_entity_emotion[line[0]+e])
            result.append((line[0], ','.join(entity), ','.join(emotion)))
        else:
            result.append((line[0], '', ''))

with open('./output/result.txt', 'w', encoding='utf-8') as f:
    for line in result:
        f.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\n')
