from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from .data_utils import dump_pkl
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_lines(path: str, col_sep: str=None) -> list:
    """
    读取文件内容。
    Params:
        path - 待读取文件的路径
        col_sep - 提取内容的条件
        
    Return:
        读取内容组成的列表
    """
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def extract_sentence(train_x_seg_path: str, train_y_seg_path: str, test_seg_path: str) -> list:
    """
    提取句子内容。
    Params:
        train_x_seg_path - train_x_seg文件的路径
        train_y_seg_path - train_y_seg文件路径
        test_seg_path - test_seg文件路径
    Return:
        返回从所有文件中提取的句子组成的列表
    """
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        ret.append(line)
    return ret


def save_sentence(lines: list, sentence_path: str):
    """
    保存句子文件到路径。
    Params:
        lines - 包含所有句子内容的列表
        sentence_path - 保存文件的路径
    """
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)


def build(train_x_seg_path: str, test_y_seg_path: str, test_seg_path: str, out_path: str=None, sentence_path: str='',
          w2v_bin_path: str="w2v.bin", min_count: int=1):
    """
    训练词向量，并保存到文件。
    Params:
        train_x_seg_path - train_x_seg 文件路径
        train_y_seg_path - train_y_seg 文件路径
        test_x_seg_path - test_x_seg 文件路径
        out_path - 保存训练好的词向量文件的位置
        sentance_path - 保存所有句子的文件的位置
        w2v_bin_path - 保存词向量2进制文件的名称
        min_count - 词最小个数
    """
    sentences = extract_sentence(train_x_seg_path, test_y_seg_path, test_seg_path)
    save_sentence(sentences, sentence_path)
    print('train w2v model...')
    # train model
    """
    通过gensim工具完成word2vec的训练，输入格式采用sentences，使用skip-gram，embedding维度256
    your code
    w2v = （one line）
    """
    w2v = Word2Vec(sentences=LineSentence(sentence_path), size=256, min_count=min_count, workers=8, sg=1, iter=50)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # test
    sim = w2v.wv.similarity('技师', '车主')
    print('技师 vs 车主 similarity score:', sim)
    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    build('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
          '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
          '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR),
          out_path='{}/datasets/word2vec.txt'.format(BASE_DIR),
          sentence_path='{}/datasets/sentences.txt'.format(BASE_DIR))
    

