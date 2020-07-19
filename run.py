from utils.preprocess import parse_data
from utils.data_reader import read_data, build_vocab, save_word_dict
from utils.build_w2v import build
from seq2seq_tf2.bin.main import main as run_seq2seq

from configs.params_config import seq2seq_params, BASE_DIR

def run(min_count: int):
    # step1: 预处理数据
    print("Step1: 进行数据预处理")
    parse_data(seq2seq_params['train_x_dir'], seq2seq_params['test_x_dir'])
    
    # step2: 构建字典文件
    print("Step2: 构建字典文件")
    lines = read_data(seq2seq_params['train_seg_x_dir'],
                      seq2seq_params['train_seg_y_dir'],
                      seq2seq_params['test_seg_x_dir'])
    vocab, _ = build_vocab(lines)
    save_word_dict(vocab, seq2seq_params['vocab_path'])
    
    # step3: 构建词向量并保存
    print("Step3:构建词向量并保存到文件")
    build(seq2seq_params['train_seg_x_dir'], # 训练数据文件路径
          seq2seq_params['train_seg_y_dir'], # 训练标签数据文件路径
          seq2seq_params['test_seg_x_dir'],  # 测试数据文件
          out_path= seq2seq_params['word2vec_output'], # 保存词向量的路径
          sentence_path='{}/datasets/sentences.txt'.format(BASE_DIR), # 将所有分词后的数据按行合并后写入文件的路径
          w2v_bin_path='{}/datasets/w2v.bin'.format(BASE_DIR), # 词向量文件保存路径和文件名
          min_count=min_count) # 词频过滤
    print("构建词向量完毕！")

def seq2seq():
    run_seq2seq()
    
if __name__ == "__main__":
    # import tensorflow as tf
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    #run(30)
    seq2seq()