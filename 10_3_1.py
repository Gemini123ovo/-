# 10.3.1 文本分类
# 代码10-1 自定义语料预处理函数
import tensorflow as tf
from collections import Counter
from tensorflow import keras
import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import MultipleLocator
import os
# 打开文件
def open_file(filename, mode='r'):
    '''
    filename：表示读取/写入的文件路径
    mode：'r' or 'w'表示读取/写入文件
    '''
    return open(filename, mode, encoding='utf-8', errors='ignore')
# 读取文件数据
def read_file(filename):
    '''
    filename：表示文件路径
    '''
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')  # 按照制表符分割字符串
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels
# 构建词汇表
def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    '''
    train_dir：训练集文件的存放路径
    vocab_dir：词汇表的存放路径
    vocab_size：词汇表的大小
    '''
    data_train, lab = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)  # 词袋
    count_pairs = counter.most_common(vocab_size - 1)  # top n
    words, temp = list(zip(*count_pairs))  # 获取key
    words = ['<PAD>'] + list(words)  # 添加一个<PAD>将所有文本pad为同一长度
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
# 读取词汇表
def read_vocab(vocab_dir):
    '''
    vocab_dir：词汇表的存放路径
    '''
    with open_file(vocab_dir) as fp:
        words = [i.strip() for i in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id
# 读取分类目录
def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # 得到类别与编号相对应的字典，分别为0-9
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id
# 将id表示的内容转换为文字
def to_words(content, words):
    '''
    content：id表示的内容
    words：文本内容
    '''
    return ''.join(words[x] for x in content)
# 将文件转换为id表示
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    '''
    filename：文件路径
    word_to_id：词汇表
    cat_to_id：类别对应的编号
    max_length：词向量的最大长度
    '''
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    # 使用Keras提供的pad_sequences将文本pad为固定长度
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    # 将标签转为独热编码（one-hot）表示
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    return x_pad, y_pad


# 代码10-2 加载数据并进行预处理

# 设置数据读取、模型、结果保存路径
base_dir = '../data/'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
save_dir = '../tmp/'
save_path = os.path.join(save_dir, 'best_validation')

# 若不存在词汇表，则重新建立词汇表
vocab_size = 5000
if not os.path.exists(vocab_dir):
    build_vocab(train_dir, vocab_dir, vocab_size)

# 读取分类目录
categories, cat_to_id = read_category()
# 读取词汇表
words, word_to_id = read_vocab(vocab_dir)
# 词汇表大小
vocab_size = len(words)

# 数据加载
seq_length = 600  # 序列长度

# 获取训练数据
x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
# 获取验证数据
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)
# 获取测试数据
x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, seq_length)

# 代码10-3 设置模型参数并构建模型


# 搭建LSTM模型
def TextRNN():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size+1, 128, input_length=600))
    # 使用LSTM的单向循环神经网络
    model.add(tf.keras.layers.LSTM(128))  
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6, axis=1))  # 标准化处理
    model.add(tf.keras.layers.Dense(256, activation='relu'))  #全连接层，激活函数为relu
    model.add(tf.keras.layers.Dropout(0.3))  # dropout正则化，随机丢弃30%的神经元，防止过拟合
    model.add(tf.keras.layers.Dense(128, activation='relu'))  #全连接层，激活函数为relu
    model.add(tf.keras.layers.Dropout(0.2))  # dropout正则化，随机丢弃20%的神经元
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 全连接层，激活函数为softmax
    return model
# 模型实例化
model = TextRNN()

# 代码10-4 模型训练

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# 训练参数设置
with strategy.scope():
    model = TextRNN()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['categorical_accuracy'])
# 模型训练
history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_val, y_val))
# 设置绘图的字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'SimHei'
# 绘制训练过程
def plot_acc_loss(history):
    '''
    history：模型训练的返回值
    '''
    plt.subplot(121)
    plt.title('准确率趋势图')
    plt.plot(range(1, 21), history.history['categorical_accuracy'], linestyle='-', color='g', label='训练集')
    plt.plot(range(1, 21), history.history['val_categorical_accuracy'], linestyle='-.', color='b', label='测试集')
    plt.legend(loc='best')  # 设置图例
    # x轴按1刻度显示
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)  
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.subplot(122)
    plt.title('损失趋势图')
    plt.plot(range(1, 21), history.history['loss'], linestyle='-', color='g', label='训练集')
    plt.plot(range(1, 21), history.history['val_loss'], linestyle='-.', color='b', label='测试集')
    plt.legend(loc='best')
    # x轴按1刻度显示
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)  
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.tight_layout()
    plt.show()
    plt.savefig("3.png")
plot_acc_loss(history)

# 代码10-5 查看模型架构并保存模型
model.summary()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.save(os.path.join(save_dir, '../tmp/my_model.h5'))
del model

# 代码10-6 模型测试

# 导入已经训练好的模型
model1 = load_model('../tmp/my_model.h5')

# 对测试集进行预测
y_pre = model1.predict(x_test)
# 计算混淆矩阵
confm = confusion_matrix(np.argmax(y_pre, axis=1), np.argmax(y_test, axis=1))
# 打印模型评价
print(classification_report(np.argmax(y_pre, axis=1), np.argmax(y_test, axis=1)))

# 混淆矩阵可视化
plt.figure(figsize=(8, 8), dpi=600)
# 设置绘图的字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'SimHei'
sns.heatmap(confm.T, square=True, annot=True,
            fmt='d', cbar=False, linewidths=.8,
            cmap='YlGnBu')
plt.xlabel('真实标签', size=14)
plt.ylabel('预测标签', size=14)
plt.xticks(np.arange(10)+0.5, categories, size=12)
plt.yticks(np.arange(10)+0.3, categories, size=12)
plt.show()
plt.savefig("1.png")