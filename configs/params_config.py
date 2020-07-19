import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


seq2seq_params = {
    "max_enc_len": 100, # help="Encoder input max sequence length", type=int encoder中输入sequence的最大长度
    "max_dec_len": 40, # help="Decoder input max sequence length", type=int)  decoder中每次解码的词的最大个数
    "max_dec_steps": 100, # help="maximum number of words of the predicted abstract", type=int) decoder后输出的词的最大数量
    "min_dec_steps": 30, # help="Minimum number of words of the predicted abstract", type=int) decoder后输出的词的最小数量
    "batch_size": 64, # help="batch size", type=int)
    "beam_size": 3, # help="beam size for beam search decoding (must be equal to batch size in decode mode)", type=int)
    "vocab_size": 30000, # help="Vocabulary size", type=int) # 读取词的个数（根据字典的值来取常用的）
    "embed_size": 256, # help="Words embeddings dimension", type=int) # 词向量的维度
    "enc_units": 512, # help="Encoder GRU cell units number", type=int) # encoder层输出的维度
    "dec_units": 512, # help="Decoder GRU cell units number", type=int) # decoder层输出的维度
    "attn_units": 256, #help="[context vector, decoder state, decoder input] feedforward result dimension this result is used to compute the attention weights", type=int)
    "learning_rate": 0.001, # help="Learning rate", type=float)
    "adagrad_init_acc": 0.1, # help="Adagrad optimizer initial accumulator value. Please refer to the Adagrad optimizer "
                             # "API documentation on tensorflow site for more details.", type=float)
    "max_grad_norm": 0.8, # help="Gradient norm above which gradients must be clipped", type=float)
    "cov_loss_wt": 0.5, # help='Weight of coverage loss (lambda in the paper).'
                        # ' If zero, then no incentive to minimize coverage loss.', type=float)

    # path
    # /ckpt/checkpoint/checkpoint
    "seq2seq_model_dir": '{}/ckpt/seq2seq'.format(BASE_DIR), # help="Model folder")
    "pgn_model_dir": '{}/ckpt/pgn'.format(BASE_DIR), # help="Model folder")
    "model_path": '', #  help="Path to a specific model", type=str)
    "train_seg_x_dir": '{}/datasets/train_set.seg_x.txt'.format(BASE_DIR), # help="train_seg_x_dir")
    "train_seg_y_dir": '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR), # help="train_seg_y_dir")
    "test_seg_x_dir": '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR), # help="test_seg_x_dir")
    "vocab_path": '{}/datasets/vocab.txt'.format(BASE_DIR), # help="Vocab path")
    "word2vec_output": '{}/datasets/word2vec.txt'.format(BASE_DIR), # help="Vocab path")
    "log_file": "",  # help="File in which to redirect console outputs", type=str)
    "test_save_dir": '{}/datasets/'.format(BASE_DIR), # help="test_save_dir")
    "test_x_dir": '{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR), # help="test_x_dir")
    "train_x_dir": '{}/datasets/AutoMaster_TrainSet.csv'.format(BASE_DIR),
    
    # others
    "steps_per_epoch": 1300, # help="max_train_steps", type=int)
    "checkpoints_save_steps": 10, # help="Save checkpoints every N steps", type=int)
    "max_steps": 10000, # help="Max number of iterations", type=int)
    "num_to_test": 10, # help="Number of examples to test", type=int)
    "max_num_to_eval": 5, # help="max_num_to_eval", type=int)
    "epochs": 20, # help="train epochs", type=int)
    
    # mode
    "mode": 'test', # help="training, eval or test options")
    "model": 'SequenceToSequence', # help="which model to be slected")
    "beam_decode": True, # help="greedy_decoder or beam_decode")
    "greedy_decode": False,
    'is_coverage': False
}