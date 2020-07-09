from utils.preprocess import parse_data
from utils.data_reader import read_data, build_vocab, save_word_dict
from utils.build_w2v import build
from seq2seq_tf2.bin.main import main as run_seq2seq

from configs.params_config import seq2seq_params, BASE_DIR

def run():
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
    build(seq2seq_params['train_seg_x_dir'],
          seq2seq_params['train_seg_y_dir'],
          seq2seq_params['test_seg_x_dir'],
          out_path= seq2seq_params['word2vec_output'],
          sentence_path='{}/datasets/sentences.txt'.format(BASE_DIR),
          w2v_bin_path='{}/datasets/w2v.bin'.format(BASE_DIR))
    print("构建词向量完毕！")

def seq2seq():
    run_seq2seq()
    
if __name__ == "__main__":
    #run()
    seq2seq()