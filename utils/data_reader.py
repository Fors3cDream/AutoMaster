from collections import defaultdict
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_word_dict(vocab: list, save_path: str):
    """
    保存字典。
    Params:
        vocab - 所有词语组成的字典
        save_path - 保存字典的文件路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def read_data(path_1: str, path_2: str, path_3: str) -> list:
    """
    读取数据。
    Params:
        path_n - 数据源路径。
    Return:
       包含所有数据源中的词语组成的列表。
    """
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        # print(f1)
        for line in f1:
            words = line.split(' ')

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items: list, sort: bool=True, min_count: int=0, lower: bool=False) -> list:
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # sort
        """
        按照字典里的词频进行排序，出现次数多的排在前面
        your code(one line)
        """
        dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)
    """
    建立项目的vocab和reverse_vocab，vocab的结构是（词，index），而reverse_vocab的结构是(index, 词)
    your code
    vocab = (one line)
    reverse_vocab = (one line)
    """
    vocab = [(w, i) for i, w in enumerate(result)]
    # 将词典中的词
    reverse_vocab = [(w[1], w[0]) for w in vocab]

    return vocab, reverse_vocab


if __name__ == '__main__':
    lines = read_data('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
                      '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
                      '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR))
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, '{}/datasets/vocab.txt'.format(BASE_DIR))
    print("保存完毕！文件位于：{}/datasets/vocab.txt".format(BASE_DIR))