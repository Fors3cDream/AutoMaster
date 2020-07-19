import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        """
        @param: vocab_size - 训练词典的大小
        @param: embedding_dim - 词嵌入维度
        """
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz # 批次数量大小
        # self.enc_units = enc_units
        self.enc_units = enc_units // 2 # encoding 元素个数
        """
        定义Embedding层，加载预训练的词向量
        your code
        """
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)
        
        # tf.keras.layers.GRU自动匹配cpu、gpu
        """
        定义单向的RNN、GRU、LSTM层
        your code
        """
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        # output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))
    
