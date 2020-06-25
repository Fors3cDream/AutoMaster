#### 问答摘要与推理 - Baidu AIStudio 竞赛代码

[竞赛及数据地址](https://aistudio.baidu.com/aistudio/competition/detail/3)

##### 文件描述
`datasets` - 保存竞赛文件及代码对数据进行处理后保存数据的文件。

`utils` - 保存代码文件。

`run.py` - 调用`utils`中相关代码按`预处理数据 -> 生成相关字典文件 -> 生成词向量并保存到文件`的顺序完成相关任务。

##### 代码功能描述：

`preprocess.py`

预处理数据，读取训练数据及测试数据，并对数据进行分词，提取训练数据中的Question列和Dialogue列数据，并连接两列数据然后进行分词处理作为训练特征。提取Report列数据进行分词处理作为训练标签数据。

生成文件`train_set.seg_x.txt` - 训练特征文件，每一行为对句子进行分词处理后的词语，`train_set.set_y.txt` - 训练标签数据，每一行为对句子进行分词处理后的词语，`test_set.seg_x.txt` - 预测数据特征，每一行为对句子进行分词处理后的词语。


`data_reader.py`

读取`preprocess.py`对竞赛数据处理后生成的三个文件，生成对应的字典，并保存到`datasets/vocab.txt`文件中。


`build_w2v.py`

调用相关函数对`preprocess.py`和`data_reader.py`对训练数据进行预处理后生成的相关文件进行处理，生成词向量并保存到文件。