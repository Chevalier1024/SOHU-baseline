#! -*- coding:utf-8 -*-

import json
import numpy as np
np.random.seed(42)
import re
from gensim.models import word2vec, KeyedVectors
from random import choice
from tqdm import tqdm
import time
import logging
from sklearn.metrics import f1_score
import pandas as pd
import jieba.posseg as pseg

file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)



train_data = json.load(open('./ner_data/train_data.json'))
dev_data = json.load(open('./ner_data/dev_data.json'))
test_data = json.load(open('./ner_data/test_data.json'))

id2char, char2id = json.load(open('./ner_data/all_chars.json'))

char_size = 128
print(len(train_data))

def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            T, S1, S2 = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d['content']
                items = {}
                for pair in d['coreEntityEmotions']:
                    try:
                        entity = re.sub('\(', '\(', pair[0])
                        entity = re.sub('\)', '\)', entity)
                        entity = re.sub('\+', '\+', entity)
                        entity = re.sub('\*', '\*', entity)

                        entityid = [i.start() for i in re.finditer(entity, text)]
                        if len(entityid) != 0:
                            items[pair[0]] = entityid
                    except Exception as e:
                        print(pair[0] + '\t' + text)
                T.append([char2id.get(c, 1) for c in text]) # 1是unk，0是padding 将文本转为ID
                # T.append(text_2_id(text)) # 1是unk，0是padding 将文本转为ID
                s1, s2 = [0] * len(text), [0] * len(text)   # s1:主体起始位置 s2:主体终止位置
                for key, value in items.items():
                    for v in value:
                        try:
                            s1[v] = 1
                            s2[v+len(key)-1] = 1
                        except Exception as e:
                            print(e)
                S1.append(s1)   # s1:主体起始位置  (batch_size, sen_len)
                S2.append(s2)   # s2:主体终止位置 (batch_size, sen_len)
                if len(T) == self.batch_size:
                    T = np.array(seq_padding(T))
                    S1 = np.array(seq_padding(S1))
                    S2 = np.array(seq_padding(S2))
                    yield [T, S1, S2], None
                    T, S1, S2 = [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
# session = tf.Session(config=config)
#
# # 设置session
# KTF.set_session(session)


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tf.gather_nd(seq, idxs)


def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = K.expand_dims(vec, 1)
    vec = K.zeros_like(seq[:, :, :1]) + vec
    return K.concatenate([seq, vec], 2)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)

def seq_avgpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.mean(seq, 1)


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


t_in = Input(shape=(None,))   # 文本
s1_in = Input(shape=(None,))    # one—hot 主体起始位置
s2_in = Input(shape=(None,))    # one-hot 客体起始位置

t, s1, s2,= t_in, s1_in, s2_in

mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(t)

t = Embedding(len(char2id)+2, char_size)(t) # 0: padding, 1: unk
t = Dropout(0.25)(t)
t = Lambda(lambda x: x[0] * x[1])([t, mask])
t = Bidirectional(CuDNNLSTM(char_size // 2, return_sequences=True))(t)

t_max = Lambda(seq_maxpool)([t, mask])
t_dim = K.int_shape(t)[-1]

h = Lambda(seq_and_vec, output_shape=(None, t_dim*2))([t, t_max])
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)

subject_model = Model(t_in, [ps1, ps2]) # 预测subject的模型

train_model = Model([t_in, s1_in, s2_in],
                    [ps1, ps2])

s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * mask) / K.sum(mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * mask) / K.sum(mask)

loss = s1_loss + s2_loss

train_model.add_loss(loss)
train_model.compile(optimizer='adam')
train_model.summary()


def extract_items(text_in):
    R = []
    # _s = text_2_id(text_in)
    _s = [char2id.get(c, 1) for c in text_in]
    _s = np.array([_s])
    _k1, _k2 = subject_model.predict(_s)
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    for i,_kk1 in enumerate(_k1):
        if _kk1 > 0.5:
            _subject = ''
            for j,_kk2 in enumerate(_k2[i:]):
                if _kk2 > 0.5:
                    entity = text_in[i: i+j+1]
                    if len(entity) <= 41 and '\n' not in entity and ',' not in entity:
                        R.append(entity)
                    break
    # if len(R) == 0:
    #     return give_me_one(text_in, _s, list(_k1), list(_k2))
    R = list(set(R))
    if len(R) > 3:
        return R[:3]
    return R

# def give_me_one(text_in, _s, k1, k2):
#     R = []
#     i = k1.index(max(k1))
#     j = k2.index(max(k2[i:]))
#     _subject = text_in[i: i+j+1]
#     _k1, _k2 = np.array([i]), np.array([i + j])
#     _e = object_model.predict([_s, _k1, _k2])
#     _e = np.argmax(_e[0])
#     R.append((_subject, id2emotion[_e]))
#     return list(set(R))


class Evaluate(Callback):
    def __init__(self, data, ground_truth):
        self.F1 = []
        self.best = 0.
        self.early_stopping = 0
        self.data = data
        self.predict = []
        self.ground_truth = ground_truth
    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('./ner_data/best_model.weights')
            self.early_stopping = 0
            output = pd.DataFrame()
            output['text'] = self.data
            output['predict'] = self.predict
            output['ground_truth'] = self.ground_truth
            output.to_csv('./output/predict_dev.csv', index=False, sep='\t')
        else:
            self.early_stopping += 1
        logger.info('epoch: %d, f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (epoch, f1, precision, recall, self.best))

        if self.early_stopping == 5:
            import sys
            logger.info('early stopping ')

            # predict
            train_model.load_weights('./ner_data/best_model.weights')
            result = []
            for i, d in tqdm(enumerate(iter(test_data))):
                R = set(extract_items(d['content']))
                sentiment = ['POS'] * len(R)
                result.append((d['newsId'], ','.join(list(R)), ','.join(sentiment)))

            with open('./output/ner.txt', 'w', encoding='utf-8') as f:
                for line in result:
                    f.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\n')

            sys.exit(0)

    def evaluate(self):
        A, B, C = 1e-10, 1e-10, 1e-10
        self.predict = []
        for i, d in tqdm(enumerate(iter(dev_data))):
            R = set(extract_items(d['content']))
            T = set([i[0] for i in d['coreEntityEmotions']])
            A += len(R & T)
            B += len(R)
            C += len(T)
            self.predict.append(R)
        return 2 * A / (B + C), A / B, A / C

data = []
ground_truth = []
for i, d in tqdm(enumerate(iter(dev_data))):
    T = set([i[0] for i in d['coreEntityEmotions']])
    data.append(d['content'])
    ground_truth.append(T)

train_D = data_generator(train_data)
evaluator = Evaluate(data ,ground_truth)
train_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=len(train_D),
                          epochs=100,
                          callbacks=[evaluator]
                          )


