#! -*- coding:utf-8 -*-

import json
from tqdm import tqdm
import codecs
import numpy as np
import random
from sklearn.model_selection import train_test_split
import re

data = []
entity = set()

with open('./data/coreEntityEmotion_train.txt', encoding='utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l.strip())
        contents = re.split(r'[\n。！？]', a['content'])
        contents.append(a['title'])
        for i in a['coreEntityEmotions']:
            sentiment_relate = []
            for content in contents:
                if i['entity'] in content:
                    sentiment_relate.append(content)
            data.append(
                {
                    'entity': i['entity'],
                    'content': ' '.join(sentiment_relate),
                    'emotion': i['emotion'],
                }
            )
            entity.add(i['entity'])

with open('./sentiment_data/user_dict.txt', 'w', encoding='utf-8') as f:
    for e in entity:
        f.write(e + ' ' + str(1000000) + '\n')

#
with codecs.open('./sentiment_data/all_sentiment_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

train_data, dev_data = train_test_split(data, random_state=2019, test_size=0.2)


with codecs.open('./sentiment_data/train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)
with codecs.open('./sentiment_data/dev_data.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)


id_content = dict()
with open('./data/coreEntityEmotion_test_stage1.txt', encoding='utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l.strip())
        content = a['title'] + '\n' + a['content']
        newsId = a['newsId']
        id_content[newsId] = content

test_data = []
with open('./output/ner.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        if len(line) == 3:
            entity = line[1].split(',')
            emotion = line[2].split(',')
            contents = re.split(r'[\n。！？]', id_content[line[0]])

            for en, em in zip(entity, emotion):
                sentiment_relate = []
                for content in contents:
                    if en in content:
                        sentiment_relate.append(content)
                test_data.append(
                    {
                        'newsId': line[0],
                        'entity': en,
                        'content': ' '.join(sentiment_relate)
                    }
                )


with codecs.open('./sentiment_data/test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)