import jieba
from jieba import posseg


# def segment_line(line):
#     tokens = jieba.cut(line, cut_all=False)
#     return " ".join(tokens)
#
#
# def process_line(line):
#     if isinstance(line, str):
#         tokens = line.split("|")
#         result = [segment_line(t) for t in tokens]
#         return " | ".join(result)


def segment(sentence: str, cut_type: str='word', pos: bool=False) -> list:
    """
    对句子进行分词操作。
    :param sentence: 需要分词的句子
    :param cut_type: 'word' use jieba.lcut; 'char' use list(sentence) 分词方式
    :param pos: enable POS 是否启用POS（词性标注）
    :return: list 分词后所有词组成的列表
    """
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)
