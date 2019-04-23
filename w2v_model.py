import os
import json
from gensim.models.word2vec import LineSentence, Word2Vec
import jieba


def func(fin, fout):
    for line in fin:
        text = list(jieba.cut(line['content']))
        fout.write(' '.join(text) + '\n')


def make_corpus():
    #print("-------------haha")
    with open('./sentiment_data/corpus.txt', 'wt', encoding='utf-8') as fout:
        train_data = json.load(open('./sentiment_data/train_data.json'))
        func(train_data, fout)
        dev_data = json.load(open('./sentiment_data/dev_data.json'))
        func(dev_data, fout)
        test_data = json.load(open('./sentiment_data/test_data.json'))
        func(test_data, fout)

if __name__ == "__main__":
    if not os.path.exists('./sentiment_data/corpus.txt'):
        make_corpus()

    sentences = LineSentence('./sentiment_data/corpus.txt')
    model = Word2Vec(sentences, sg=1, size=128, workers=4, iter=8, negative=8, min_count=2)
    word_vectors = model.wv
    word_vectors.save_word2vec_format('./w2v/word2vec.txt', fvocab='./w2v/vocab.txt')