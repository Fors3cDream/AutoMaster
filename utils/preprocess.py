import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba
from .tokenizer import segment
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REMOVE_WORDS = ['|', '[', ']', '语音', '图片', ' ', '：']

def read_stopwords(path: str) -> set:
    """
    读取停止词，返回停止词组成的列表。
    Params:
        path - 停止词文件路径。
    Return:
        所有停止词组成的列表。
    """
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


def remove_words(words_list: list) -> list:
    """
    移除停止词。
    Params:
        words_list -  包含所有词语的列表。
    Return:
        移除停止词后的词语组成的列表。
    """
    #words_list = [word for word in words_list if word not in REMOVE_WORDS]
    REMOVE_WORDS = list(read_stopwords(os.path.join(BASE_DIR, 'datasets/stopwords.txt')))
    words_list = [word.strip() for word in words_list if word not in REMOVE_WORDS]
    return words_list


def parse_data(train_path: str, test_path: str):
    """
    处理训练数据和测试数据。
    Params:
        train_path - 训练数据文件路径
        test_path - 测试数据文件路径
    """
    # 读取训练数据
    train_df = pd.read_csv(train_path, encoding='utf-8')
    # 丢弃 Report 中为空的数据，直接替换原data frame
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    # 填充空数据
    train_df.fillna('', inplace=True)
    # 训练数据中的x为，Question列和Dialogue列数据的组合
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    print('train_x is ', len(train_x))
    # 对train_x数据进行处理，处理函数为 preprocess_sentence
    train_x = train_x.apply(preprocess_sentence)
    print('train_x is ', len(train_x))
    
    # train_y数据为 Report列数据
    train_y = train_df.Report
    print('train_y is ', len(train_y))
    # 同样使用preprocess_sentence函数对train_y数据进行处理
    train_y = train_y.apply(preprocess_sentence)
    print('train_y is ', len(train_y))
    # if 'Report' in train_df.columns:
        # train_y = train_df.Report
        # print('train_y is ', len(train_y))

    # 读取测试数据
    test_df = pd.read_csv(test_path, encoding='utf-8')
    # 填充空数据
    test_df.fillna('', inplace=True)
    # 取Question列和Dialogue列组成 test_x 数据
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    # 使用preprocess_sentence函数对数据进行处理
    test_x = test_x.apply(preprocess_sentence)
    print('test_x is ', len(test_x))
    #test_y = []
    # 保存生成的数据到文件
    train_x.to_csv('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)
    train_y.to_csv('{}/datasets/train_set.seg_y.txt'.format(BASE_DIR), index=None, header=False)
    test_x.to_csv('{}/datasets/test_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)


def preprocess_sentence(sentence: str) -> str:
    """
    处理句子。
    Params:
        sentence - 待处理的句子
    Return:
        经过分词处理和去除停止词的词语组成的长句子，词语之间用空格隔开。
    """
    # 调用 segment 函数对句子进行分词处理
    seg_list = segment(sentence.strip(), cut_type='word')
    # 调用 remove_words 对分词后的词语列表进行去除停止词处理
    seg_list = remove_words(seg_list)
    # 组合词语为一个字符串，以空格为间隔符号。
    seg_line = ' '.join(seg_list)
    return seg_line


if __name__ == '__main__':
    # 需要更换成自己数据的存储地址
    parse_data('{}/datasets/AutoMaster_TrainSet.csv'.format(BASE_DIR),
               '{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR))


