import os
from utils.preprocess import parse_data
from utils.data_reader import read_data, build_vocab, save_word_dict
from utils.build_w2v import build


BASE_DIR = os.path.dirname(__file__)

def run():
    # step1: 预处理数据
    print("Step1: 进行数据预处理")
    # parse_data('{}/datasets/AutoMaster_TrainSet.csv'.format(BASE_DIR),'{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR))
    
    # step2: 构建字典文件
    print("Step2: 构建字典文件")
    # lines = read_data('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
    #                   '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
    #                   '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR))
    # vocab, _ = build_vocab(lines)
    # save_word_dict(vocab, '{}/datasets/vocab.txt'.format(BASE_DIR))
    
    # step3: 构建词向量并保存
    print("Step3:构建词向量并保存到文件")
    build('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
          '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
          '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR),
          out_path='{}/datasets/word2vec.txt'.format(BASE_DIR),
          sentence_path='{}/datasets/sentences.txt'.format(BASE_DIR),
          w2v_bin_path='{}/datasets/w2v.bin'.format(BASE_DIR))
    print("构建词向量完毕！")
    
    
if __name__ == "__main__":
    run()