#! -*- coding:utf-8 -*-

import json
from tqdm import tqdm
import codecs
import numpy as np
import random
from sklearn.model_selection import train_test_split
import re

emotion = set()
chars = {}
data = []
min_count = 2

with open('./data/coreEntityEmotion_train.txt', encoding='utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l.strip())
        data.append(
            {
                'content': a['title'] + '\n' + a['content'],
                'coreEntityEmotions': [(i['entity'], i['emotion']) for i in a['coreEntityEmotions']],
            }
        )
        for c in a['content']:
            chars[c] = chars.get(c, 0) + 1
        for c in a['title']:
            chars[c] = chars.get(c, 0) + 1

        for c in a['coreEntityEmotions']:
            emotion.add(c['emotion'])


id2emotion = {i:j for i,j in enumerate(emotion)}
emotion2id = {j:i for i,j in id2emotion.items()}

with open('./process_data/emotion.json', 'w', encoding='utf-8') as f:
    json.dump([id2emotion, emotion2id], f, indent=4, ensure_ascii=False)


with codecs.open('./process_data/all_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

train_data, test_data = train_test_split(data, random_state=2019, test_size=0.2)


# new_train_data = []
# for item in train_data:
#     contents = re.split(r'[\n。！？]', item['content'])
#     for text in contents:
#         if len(text) < 5:
#             continue
#         new_train_data.append(
#             {
#                 'content': text,
#                 'coreEntityEmotions': item['coreEntityEmotions'],
#             }
#         )

with codecs.open('./process_data/train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)
with codecs.open('./process_data/dev_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)

test_data = []
with open('./data/coreEntityEmotion_test_stage1.txt', encoding='utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l.strip())
        test_data.append(
            {
                'newsId': a['newsId'],
                'content': a['title'] + '\n' + a['content'],
                # 'coreEntityEmotions': [(i['entity'], i['emotion']) for i in a['coreEntityEmotions']],
            }
        )
        for c in a['content']:
            chars[c] = chars.get(c, 0) + 1
        for c in a['title']:
            chars[c] = chars.get(c, 0) + 1

with codecs.open('./process_data/test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)


with codecs.open('./process_data/all_chars.json', 'w', encoding='utf-8') as f:
    chars = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+2:j for i,j in enumerate(chars)} # padding: 0, unk: 1
    char2id = {j:i for i,j in id2char.items()}
    json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)
