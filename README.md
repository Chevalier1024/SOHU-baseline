# SOHU-baseline
# 搜狐校园算法大赛baseline
  [比赛网址]()

## 代码运行环境：
   * python 3.6
   * Keras 2.2.4
   * tqdm
   * jieba
   * tensorflow-gpu 1.12.0

## 整体思路：
* 采用pipeline的方式，将这个任务拆为两个子任务，先预测aspect，根据aspect预测情感极性（ABSA），这两个子任务都使用深度学习模型解决
* aspect预测采用指针标注的方式解决，标注aspect的头和尾，思路参考苏神在百度信息抽取的baseline
* 基于aspect的情感分析是一个多分类问题，首先根据分隔符将文本拆成多段，然后拼接aspect出现过的文本，再进行三分类

## 代码框架：
* baseline/: 官方的baseline
* data/: 比赛的原始数据
* log/: 日志输出文件
* ner/: 训练ner模型的相关数据
* output/:最终结果文件
* sentiment_data/:训练情感分类模型的相关数据
* w2v/: 词向量
* analysis.py: 数据分析
* ner.py: ner模型
* ner_corpus.py: 生成ner训练数据
* sentiment_corpus.py: 生成情感分类训练数据
* sentiment_model.py: 情感分类模型
* w2v_model.py: 训练词向量

** 执行顺序：
```
1.将原始数据放入data文件夹
2.运行ner_corpus.py生成ner模型训练语料
3.运行ner.py
4.运行sentiment_corpus.py
5.运行w2v_model.py
6.运行sentiment_model.py
7.提交
```

** 提交结果：
* ner.txt: 0.306726656807222
* result.txt: 0.314122307927767

** 相关资料：
* https://github.com/wangbin4317/BDCI_Car_2018-master 
* https://github.com/binzhouchn/capsule-pytorch                
* https://github.com/idorce/sentiment-analysis-ccf-bdci        
* https://github.com/yw411/aspect_sentiment_classification        
* https://github.com/songyouwei/SA-DL                                     
* https://github.com/pengshuang/AI-Comp

** 给start的都能进前排哦~O(∩_∩)O哈哈~






