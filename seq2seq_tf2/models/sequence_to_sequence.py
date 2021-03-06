import tensorflow as tf
from seq2seq_tf2.encoders import rnn_encoder
from seq2seq_tf2.decoders import rnn_decoder
from utils.data_utils import load_word2vec
import time


class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super(SequenceToSequence, self).__init__()
        self.embedding_matrix = load_word2vec(params) # 词向量矩阵
        self.params = params
        self.encoder = rnn_encoder.Encoder(params["vocab_size"], # 加载字典中词的数量
                                           params["embed_size"], # 词嵌入向量大小
                                           params["enc_units"], # encoder 单元个数
                                           params["batch_size"], # 分批次加载数据的大小
                                           self.embedding_matrix)
        self.attention = rnn_decoder.BahdanauAttention(params["attn_units"]) # attention 单元个数
        self.decoder = rnn_decoder.Decoder(params["vocab_size"],
                                           params["embed_size"],
                                           params["dec_units"], # decoder 单元个数
                                           params["batch_size"],
                                           self.embedding_matrix)

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        # [batch_sz, max_train_x, enc_units], [batch_sz, enc_units]
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden
    
    def call(self, enc_output, dec_inp, dec_hidden, dec_tar):
        predictions = []
        attentions = []
        context_vector, _ = self.attention(dec_hidden,  # shape=(16, 256)
                                           enc_output) # shape=(16, 200, 256)
        # 第一次输入
        # dec_inp = tf.expand_dims(dec_inp[:, 0], axis=1)
        for t in range(dec_tar.shape[1]): # 50
            # Teachering Forcing
            """
            应用decoder来一步一步预测生成词语概论分布
            your code
            如：xxx = self.decoder(), 采用Teachering Forcing方法
            """
            
            # print('dec_inp shape: {}'.format(dec_inp.shape))
            _, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:,t], axis=1), dec_hidden, enc_output, context_vector)
            
            # Teachering Forcing
            #dec_inp = tf.expand_dims(dec_tar[:, t], axis=1)
            
            context_vector, attn_dist = self.attention(dec_hidden, enc_output)
                      
            predictions.append(pred)
            attentions.append(attn_dist)

        return tf.stack(predictions, 1), dec_hidden