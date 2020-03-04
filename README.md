# Active-Learning-for-NER
本项目在模型BiLSTM-CRF上实现了一部分基于不确定性的主动学习算法，其中实现的样本选择策略包括MNLP、LC、TTE、TE。
## 数据集
数据集包括中文和英文两类命名实体识别数据集。

中文数据集使用的是细粒度的CLUENER数据集，https://github.com/CLUEbenchmark/CLUENER2020

英文数据集使用的是CoNLL-2003

## 预训练词向量
中文预训练词向量来自Chinese-Word-Vectors中使用人名日报数据集训练出的词向量，https://github.com/Embedding/Chinese-Word-Vectors

英文预训练词向量使用glove.840B.300d

## 模型结构
BiLSTM-CRF，见文件networks.py

## 实现的样本选择策略
MNLP[1]、LC[2]、TTE[2]、TE[2]

## 使用方法
`$python run.py`

## Prerequisites
* python 3.6
* tensorflow 1.13.1
* numpy 1.16.5
* keras-preprocessing 1.1.0

## 本项目所使用的主动学习流程
1. 使用部分样本训练最初的模型；
2. 使用样本选择策略选择样本；
3. 将选择好的样本加入到训练集中；
4. 重复1-3过程，直到总样本数量达到指定值(由参数total_num决定)；
5. 使用最终选择出的样本重新训练整个模型；

## 参考文献
[1] Deep active learning for named entity recognition

[2] An Analysis of Active Learning Strategies for Sequence Labeling Tasks
