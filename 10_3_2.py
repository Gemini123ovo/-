# 10.3.2 情感分析
# 代码10-7 读取语料数据
import pandas as pd
from tensorflow.keras.preprocessing import sequence
import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras import Input
import time
from sklearn import metrics
import os  # 新增：导入os模块，用于创建保存目录

# ====================== 新增：设置保存路径 ======================
save_dir = '../tmp/'  # 模型/词表保存目录（和文本分类模型同目录）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 若目录不存在则创建
model_save_path = os.path.join(save_dir, 'sentiment_model.h5')  # 情感模型保存路径
dict_save_path = os.path.join(save_dir, 'sentiment_dict.pkl')  # 词表保存路径

# 读取正负情感语料
neg = pd.read_excel('../data/neg.xls', header=None, index_col=None)
pos = pd.read_excel('../data/pos.xls', header=None, index_col=None)

# 给训练语料贴标签
pos['mark'] = 1
neg['mark'] = 0

# 代码10-8 词语向量化
# 分词
cut_word = lambda x: list(jieba.cut(x))  # 定义分词函数
pn_all = pd.concat([pos, neg], ignore_index=True)  # 合并正负情感语料
# 先确保所有输入都是字符串类型
pn_all[0] = pn_all[0].astype(str)

# 然后应用分词
cut_word = lambda x: list(jieba.cut(x))  # 定义分词函数
pn_all['words'] = pn_all[0].apply(cut_word)  # 对情感语料分词
comment = pd.read_excel('../data/sum.xls')  # 读入评论内容,增加语料
comment = comment[comment['rateContent'].notnull()]  # 仅读取非空评论
comment['words'] = comment['rateContent'].apply(cut_word)  # 对评论语料分词
pn_comment = pd.concat([pn_all['words'], comment['words']], ignore_index=True)  # 合并所有的数据

# 正负情感评论词语向量化
w = [] 
for i in pn_comment:
    w.extend(i)    
dicts = pd.DataFrame(pd.Series(w).value_counts())  # 建立统计词典
del w, pn_comment  # 删除临时文件 w，d2v_train
dicts['id'] = list(range(1, len(dicts)+1))
get_sent = lambda x: list(dicts['id'][x])
pn_all['sent'] = pn_all['words'].apply(get_sent)

# 评论词语向量标准化，对样本进行padding填充和truncating修剪
maxlen = 50  # 设置评论词语最大长度
pn_all['sent'] = list(sequence.pad_sequences(pn_all['sent'], maxlen=maxlen))  # 正负情感评论词语向量化

# 训练集、测试集
x_all = np.array(list(pn_all['sent']))
y_all = np.array(list(pn_all['mark']))
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.25)

print('训练集的特征数据形状为：', x_train.shape)
print('训练集的标签数据形状为：', y_train.shape)
print('测试集的特征数据形状为：', x_test.shape)
print('测试集的标签数据形状为：', y_test.shape)
print('训练集的特征数据为：\n', x_train)

# 代码10-9 模型构建
# 搭建LSTM模型
model = Sequential()
model.add(Input(shape=(50,)))
model.add(Embedding(len(dicts)+1, 256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

# 代码10-10 模型训练
# 设置超参
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
timeA = time.time()
model.fit(x_train, y_train, batch_size=16, epochs=10) 
timeB = time.time()
print('time cost: ', int(timeB-timeA))

# ====================== 新增：保存模型和词表 ======================
# 1. 保存训练好的情感分析模型（H5格式）
model.save(model_save_path)
print(f'情感分析模型已保存至：{model_save_path}')

# 2. 保存词表（dicts）为pickle格式（方便后续加载）
dicts.to_pickle(dict_save_path)
print(f'情感分析词表已保存至：{dict_save_path}')

# 代码10-11 模型测试
# 使用测试数据进行模型测试
y_pred = model.predict(x_test).round().astype(int)
# 模型评价
acc = metrics.accuracy_score(y_test, y_pred)
print('测试集的准确率为：', acc)
print('精确率，召回率，F1值分别为：')
print(metrics.classification_report(y_test, y_pred))
print('混淆矩阵为：')
cm = metrics.confusion_matrix(y_test, y_pred)  # 混淆矩阵
print(cm)