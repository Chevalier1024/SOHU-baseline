from tqdm import tqdm
import json
import pandas as pd
from matplotlib import pyplot as plt
import jieba
import pandas as pd
from tqdm import tqdm

# 统计最大句长，数据样本数
# max_length = 0
# data_nums = 0
# with open('./data/coreEntityEmotion_train.txt', encoding='utf-8') as f:
#     for l in tqdm(f):
#         a = json.loads(l)
#         for i in a['coreEntityEmotions']:
#             max_length = max(max_length, len(i['entity']))
#         data_nums += 1
# print(max_length)
# print(data_nums)
'''
训练集 最大句长：18015
      样本数：40000
测试集 最大句长：19288
      样本数：40000
'''

# 分别统计1个实体、2个实体、3个实体数
# entity_num = dict()
# with open('./data/coreEntityEmotion_train.txt', encoding='utf-8') as f:
#     for l in tqdm(f):
#         a = json.loads(l)
#         key = len(a['coreEntityEmotions'])
#         if key not in entity_num:
#             entity_num[key] = 1
#         else:
#             entity_num[key] += 1
# print(entity_num)
'''
{3: 16438, 2: 14170, 1: 9306, 4: 63, 6: 5, 8: 1, 5: 16, 7: 1}
'''

# 统计情感分布
# emotion = {'POS': 1, 'NEG': -1, 'NORM': 0}
# emotion_num = {-1: 0, 0: 0, 1: 0}
# with open('./data/coreEntityEmotion_train.txt', encoding='utf-8') as f:
#     for l in tqdm(f):
#         a = json.loads(l)
#         for i in a['coreEntityEmotions']:
#             emotion_num[emotion[i['emotion']]] += 1
#
# print(emotion_num)
'''
{-1: 11029, 0: 33137, 1: 43171}
'''

# 查看实体是否都在字典中出现
# nerDict = []
# with open('./models/nerDict.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         nerDict.append(line.strip())
# not_exist = []
# with open('./data/coreEntityEmotion_train.txt', encoding='utf-8') as f:
#     for l in tqdm(f):
#         a = json.loads(l)
#         for i in a['coreEntityEmotions']:
#             if i['entity'] not in nerDict:
#                 not_exist.append(i['entity'])
#
# print(len(not_exist))
# print(not_exist)
'''
11492
'''


# train = []
# with open('./data/coreEntityEmotion_train.txt', encoding='utf-8') as f:
#     for l in tqdm(f):
#         a = json.loads(l)
#         train.append(len(a['content']))
#
# length = pd.DataFrame()
# length['length'] = train
# print(length.describe())
# print(sum(train) / len(train))
'''
             length
count  40000.000000
mean    1293.505175
std     1163.674072
min       46.000000
25%      601.000000
50%      945.000000
75%     1560.000000
max    18015.000000
'''

# 分别统计实体在文本中出现的次数
# entity_num = []
# with open('./data/coreEntityEmotion_train.txt', encoding='utf-8') as f:
#     for l in tqdm(f):
#         a = json.loads(l)
#         content = a['title'] + '\n' + a['content']
#         info = dict()
#         for pair in a['coreEntityEmotions']:
#             info[pair['entity']] = content.count(pair['entity'])
#             if content.count(pair['entity']) == 0:
#                 print(content)
#         entity_num.append(info)
#
# with open('./process_data/entity_num.txt', 'w', encoding='utf-8') as f:
#     for pair in entity_num:
#         info = ''
#         for key, value in pair.items():
#             info += key + ': ' + str(value) + '\t'
#         f.write(info + '\n')

# sentence length of train data
train_data = json.load(open('./sentiment_data/train_data.json'))
length = []
for line in tqdm(train_data):
    length.append(len(list(jieba.cut(line['content']))))

l = pd.DataFrame()
l['length'] = length
print(l['length'].describe())
print(len(l[l['length'] <= 500]) / len(l))
print(len(l[l['length'] <= 600]) / len(l))
print(len(l[l['length'] <= 650]) / len(l))


