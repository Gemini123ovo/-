import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog
import threading
import re
import io
import tensorflow as tf
import numpy as np
import jieba
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn import metrics
import os
import sys
# -------------------------- 新增：导入API调用所需库 --------------------------
import requests
import json

# -------------------------- 新增：无图形化环境检测 & 适配 --------------------------
def is_display_available():
    """检测是否有图形化显示环境"""
    return os.environ.get('DISPLAY') is not None

# 禁用tkinter弹窗（无图形化时）
if not is_display_available():
    # 替换messagebox为控制台打印
    class DummyMessageBox:
        @staticmethod
        def showwarning(title, msg):
            print(f"【警告】{title}: {msg}")
        @staticmethod
        def showinfo(title, msg):
            print(f"【信息】{title}: {msg}")
    messagebox = DummyMessageBox()

# -------------------------- 全局配置 & 模型加载（适配你的代码路径） --------------------------
# 配置文件路径（需根据你的实际路径调整）
BASE_DIR = '../data/'
VOCAB_DIR = os.path.join(BASE_DIR, 'cnews.vocab.txt')
MODEL_DIR = '../tmp/'
TRANSLATE_DATA_PATH = '../data/en-ch.txt'
SENTIMENT_DICT_PATH = os.path.join(MODEL_DIR, 'sentiment_dict.pkl')

# -------------------------- 新增：豆包API配置 --------------------------
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DOUBAO_API_KEY = "c6cab858-8561-4ebe-9c83-78b3b9604560"  # 建议后续移到配置文件中
DOUBAO_MODEL = "doubao-seed-1-6-250615"

# 预加载模型（可根据实际训练后的模型路径修改）
model_loaded = True
text_classify_model = None
sentiment_model = None
translate_encoder = None
translate_decoder = None
inp_lang = None  # 机器翻译用：输入语言tokenizer
targ_lang = None  # 机器翻译用：目标语言tokenizer
max_length_targ = 0  # 机器翻译目标句子最大长度
max_length_inp = 0  # 机器翻译输入句子最大长度
units = 1024  # 机器翻译模型神经元数量（和10_4.py保持一致）

try:
    # 1. 加载文本分类模型
    text_classify_model = load_model(os.path.join(MODEL_DIR, 'my_model.h5'))
    
    # 2. 加载情感分析模型 & 词典
    sentiment_model = load_model(os.path.join(MODEL_DIR, 'sentiment_model.h5'))
    # 加载情感分析词表（增加容错）
    if os.path.exists(SENTIMENT_DICT_PATH):
        sentiment_dict = pd.read_pickle(SENTIMENT_DICT_PATH)
    else:
        raise FileNotFoundError(f"情感分析词表文件不存在：{SENTIMENT_DICT_PATH}")

    # 3. 加载机器翻译模型 & tokenizer（适配10_4.py逻辑）
    from tensorflow.train import Checkpoint
    checkpoint_dir = '../tmp/training_checkpoints'
    
    # 重建编码器/解码器（和10_4.py结构一致）
    class Encoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
            super(Encoder, self).__init__()
            self.batch_sz = batch_sz
            self.enc_units = enc_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        def call(self, x, hidden):
            x = self.embedding(x)
            output, state = self.gru(x, initial_state=hidden)
            return output, state
        def initialize_hidden_state(self):
            return tf.zeros((self.batch_sz, self.enc_units))
    
    class BahdanauAttention(tf.keras.layers.Layer):
        def __init__(self, units):
            super(BahdanauAttention, self).__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)
        def call(self, query, values):
            hidden_with_time_axis = tf.expand_dims(query, 1)
            score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector, attention_weights
    
    class Decoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
            super(Decoder, self).__init__()
            self.batch_sz = batch_sz
            self.dec_units = dec_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
            self.fc = tf.keras.layers.Dense(vocab_size)
            self.attention = BahdanauAttention(self.dec_units)
        def call(self, x, hidden, enc_output):
            context_vector, attention_weights = self.attention(hidden, enc_output)
            x = self.embedding(x)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            output, state = self.gru(x)
            output = tf.reshape(output, (-1, output.shape[2]))
            x = self.fc(output)
            return x, state, attention_weights
    
    # 加载机器翻译语料预处理后的tokenizer和参数（需先运行10_4.py生成）
    num_examples = 2000
    
    # 复用10_4.py的预处理函数
    def preprocess_sentence(w):
        w = re.sub(r'([?.!,])', r' \1 ', w)
        w = re.sub(r"[' ']+", ' ', w)
        w = '<start> ' + w + ' <end>'
        return w
    
    def create_dataset(path, num_examples):
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
        return zip(*word_pairs)
    
    def max_length(tensor):
        return max(len(t) for t in tensor)
    
    def tokenize(lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        return tensor, lang_tokenizer
    
    def load_dataset(path, num_examples=None):
        targ_lang, inp_lang = create_dataset(path, num_examples)
        input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
    
    # 加载tokenizer和长度参数
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(TRANSLATE_DATA_PATH, num_examples)
    max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1
    embedding_dim = 256
    BATCH_SIZE = 64
    
    # 重建编码器/解码器并加载权重
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    translate_encoder = encoder
    translate_decoder = decoder
    
except Exception as e:
    model_loaded = False
    # 无图形化时自动打印到控制台
    messagebox.showwarning("模型加载提示", f"部分模型加载失败：{str(e)}\n请先运行训练脚本生成模型文件")

# -------------------------- 核心功能函数（适配你的代码逻辑） --------------------------
# 1. 文本分类（适配10_3_1.py逻辑）
def text_classification(text):
    """文本分类：体育/财经/房产/家居/教育/科技/时尚/时政/游戏/娱乐"""
    if not text.strip():
        return "请输入有效文本！"
    if text_classify_model is None:
        return "文本分类模型未加载！请先运行10_3_1.py训练模型"
    
    # 适配10_3_1.py的预处理逻辑
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # 读取词汇表（简化版，完整逻辑参考10_3_1.py）
    try:
        with open(VOCAB_DIR, 'r', encoding='utf-8') as f:
            words = [i.strip() for i in f.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
    except Exception as e:
        return f"词汇表加载失败：{str(e)}！请先运行10_3_1.py生成cnews.vocab.txt"
    
    # 文本转ID + 填充
    data_id = [word_to_id[x] for x in list(text) if x in word_to_id]
    x_pad = tf.keras.preprocessing.sequence.pad_sequences([data_id], maxlen=600)
    
    # 预测
    pred = text_classify_model.predict(x_pad, verbose=0)
    pred_label = categories[np.argmax(pred)]
    return f"分类结果：{pred_label}\n置信度：{np.max(pred):.4f}"

# 2. 情感分析（适配10_3_2.py逻辑）
def preprocess_sentiment(text):
    """情感分析文本预处理"""
    # 1. 分词（和训练时一致）
    words = list(jieba.cut(str(text)))  # 确保text是字符串
    # 2. 转为ID序列（不在词表中的词映射为0）
    sent = [sentiment_dict['id'][w] if w in sentiment_dict['id'] else 0 for w in words]
    # 3. 填充/截断至固定长度（和训练时一致：maxlen=50）
    sent = tf.keras.preprocessing.sequence.pad_sequences([sent], maxlen=50)
    return sent

def sentiment_analysis(text):
    """情感分析：正面/负面"""
    if not text.strip():
        return "请输入有效文本！"
    if sentiment_model is None:
        return "情感分析模型未加载！请先运行10_3_2.py训练模型"
    
    try:
        # 预处理文本
        processed_text = preprocess_sentiment(text)
        # 预测
        pred = sentiment_model.predict(processed_text, verbose=0)
        # 判断情感倾向（0=负面，1=正面）
        sentiment = "正面" if pred[0][0] > 0.5 else "负面"
        confidence = pred[0][0] if sentiment == "正面" else 1 - pred[0][0]
        return f"情感分析结果：{sentiment}\n置信度：{confidence:.4f}"
    except Exception as e:
        return f"情感分析出错：{str(e)}"

# 3. 机器翻译（适配10_4.py逻辑，完整实现）
def machine_translation(text):
    """机器翻译：中文→英文（基于Seq2Seq）"""
    if not text.strip():
        return "请输入有效文本！"
    if translate_encoder is None or translate_decoder is None:
        return "机器翻译模型未加载！请先运行10_4.py训练模型"
    
    def evaluate(sentence):
        attention_plot = np.zeros((max_length_targ, max_length_inp))
        sentence = preprocess_sentence(sentence)
        inputs = [inp_lang.word_index[i] for i in sentence.split(' ') if i in inp_lang.word_index]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
        inputs = tf.convert_to_tensor(inputs)
        
        result = ''
        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = translate_encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
        
        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = translate_decoder(dec_input, dec_hidden, enc_out)
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += targ_lang.index_word[predicted_id] + ' '
            if targ_lang.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot
            dec_input = tf.expand_dims([predicted_id], 0)
        return result, sentence, attention_plot
    
    try:
        result, _, _ = evaluate(text)
        # 清理结果（移除<start>/<end>标记）
        result = result.replace('<start>', '').replace('<end>', '').strip()
        return f"待翻译文本：{text}\n翻译结果：{result}"
    except Exception as e:
        return f"机器翻译出错：{str(e)}"

# -------------------------- 新增：豆包API调用功能 --------------------------
def doubao_api_chat(text_query, image_url=None):
    """
    调用豆包API进行图文问答
    :param text_query: 文本问题
    :param image_url: 图片URL（可选）
    :return: 回答结果字符串
    """
    if not text_query.strip():
        return "请输入有效的问题！"
    
    # 构建请求消息体
    messages = []
    content_items = []
    
    # 添加图片（如果有）
    if image_url and image_url.strip():
        content_items.append({
            "type": "image_url",
            "image_url": {
                "url": image_url.strip()
            }
        })
    
    # 添加文本问题
    content_items.append({
        "type": "text",
        "text": text_query.strip()
    })
    
    # 构建完整消息
    messages.append({
        "role": "user",
        "content": content_items
    })
    
    # 构建请求参数
    payload = {
        "model": DOUBAO_MODEL,
        "messages": messages
    }
    
    # 设置请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DOUBAO_API_KEY}"
    }
    
    try:
        # 发送请求
        response = requests.post(
            DOUBAO_API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=30  # 设置超时时间
        )
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析响应
        response_data = response.json()
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            answer = response_data["choices"][0]["message"]["content"]
            return f"豆包回答：\n{answer}"
        else:
            return f"API响应无有效内容：{json.dumps(response_data, ensure_ascii=False)}"
    
    except requests.exceptions.Timeout:
        return "请求超时！请检查网络或稍后重试。"
    except requests.exceptions.ConnectionError:
        return "网络连接错误！请检查网络状态。"
    except requests.exceptions.HTTPError as e:
        return f"HTTP请求错误：{str(e)}\n响应内容：{response.text if 'response' in locals() else '无'}"
    except Exception as e:
        return f"豆包API调用出错：{str(e)}"

# -------------------------- GUI界面设计（仅在有图形化时加载） --------------------------
class QASystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("文本分析问答系统（含豆包API）")
        self.root.geometry("1000x700")  # 增大窗口以适配新增功能
        self.root.resizable(False, False)

        # 1. 顶部功能选择栏
        self.func_var = tk.StringVar(value="text_classify")
        func_frame = ttk.LabelFrame(root, text="功能选择", padding=10)
        func_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Radiobutton(func_frame, text="文本分类（体育/财经等）", variable=self.func_var, 
                        value="text_classify").grid(row=0, column=0, padx=10)
        ttk.Radiobutton(func_frame, text="情感分析（正面/负面）", variable=self.func_var, 
                        value="sentiment").grid(row=0, column=1, padx=10)
        ttk.Radiobutton(func_frame, text="机器翻译（中→英）", variable=self.func_var, 
                        value="translation").grid(row=0, column=2, padx=10)
        # -------------------------- 新增：豆包API功能选择按钮 --------------------------
        ttk.Radiobutton(func_frame, text="豆包图文问答", variable=self.func_var, 
                        value="doubao_api").grid(row=0, column=3, padx=10)

        # -------------------------- 新增：图片URL输入框（豆包API专用） --------------------------
        image_frame = ttk.LabelFrame(root, text="图片URL（豆包图文问答专用）", padding=10)
        image_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.image_url_var = tk.StringVar()
        ttk.Entry(image_frame, textvariable=self.image_url_var, width=80, font=("SimHei", 11)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(image_frame, text="清空URL", command=self.clear_image_url).pack(side=tk.RIGHT, padx=5)

        # 2. 输入区域
        input_frame = ttk.LabelFrame(root, text="输入文本", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        self.input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=8, font=("SimHei", 12))
        self.input_text.pack(fill=tk.BOTH, expand=True)

        # 3. 按钮区域
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=20, pady=5)

        ttk.Button(btn_frame, text="开始分析", command=self.run_analysis).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="清空输入", command=self.clear_input).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="关于", command=self.show_about).pack(side=tk.RIGHT, padx=10)

        # 4. 输出区域
        output_frame = ttk.LabelFrame(root, text="分析结果", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=12, font=("SimHei", 12))
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)  # 只读

    # -------------------------- 新增：清空图片URL --------------------------
    def clear_image_url(self):
        self.image_url_var.set("")

    # 清空输入
    def clear_input(self):
        self.input_text.delete(1.0, tk.END)
        self.clear_image_url()
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)

    # 关于弹窗
    def show_about(self):
        messagebox.showinfo("关于", "文本分析问答系统\n整合功能：\n1. 文本分类（10类）\n2. 情感分析（正负）\n3. 机器翻译（中→英）\n4. 豆包图文问答（API调用）")

    # 运行分析（子线程避免界面卡顿）
    def run_analysis(self):
        input_content = self.input_text.get(1.0, tk.END).strip()
        func_type = self.func_var.get()
        
        # 豆包API需要检查至少有问题文本
        if func_type == "doubao_api" and not input_content:
            messagebox.showwarning("提示", "请输入要提问的问题！")
            return
        
        # 其他功能检查输入
        if func_type != "doubao_api" and not input_content:
            messagebox.showwarning("提示", "请输入文本！")
            return
        
        # 清空输出
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "正在分析，请稍候...\n")
        self.output_text.config(state=tk.DISABLED)

        # 获取图片URL（仅豆包API使用）
        image_url = self.image_url_var.get() if func_type == "doubao_api" else None
        
        # 子线程执行分析
        thread = threading.Thread(target=self.analysis_worker, args=(input_content, func_type, image_url))
        thread.daemon = True
        thread.start()

    # 分析工作线程（新增image_url参数）
    def analysis_worker(self, input_content, func_type, image_url=None):
        try:
            if func_type == "text_classify":
                result = text_classification(input_content)
            elif func_type == "sentiment":
                result = sentiment_analysis(input_content)
            elif func_type == "translation":
                result = machine_translation(input_content)
            # -------------------------- 新增：豆包API调用逻辑 --------------------------
            elif func_type == "doubao_api":
                result = doubao_api_chat(input_content, image_url)
            else:
                result = "无效功能选择！"
        except Exception as e:
            result = f"分析出错：{str(e)}"
        
        # 更新输出（线程安全）
        self.root.after(0, self.update_output, result)

    # 更新输出文本
    def update_output(self, result):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, result)
        self.output_text.config(state=tk.DISABLED)

# -------------------------- 主程序入口（适配无图形化环境） --------------------------
if __name__ == "__main__":
    # 有图形化界面时启动GUI，无则提示控制台模式
    if is_display_available():
        root = tk.Tk()
        app = QASystemGUI(root)
        root.mainloop()
    else:
        print("="*50)
        print("当前环境无图形化界面，进入控制台模式！")
        print("支持功能：1.文本分类 2.情感分析 3.机器翻译 4.豆包图文问答")
        print("输入 'exit' 退出程序")
        print("="*50)
        
        while True:
            try:
                # 控制台交互逻辑
                print("\n请选择功能：")
                print("1 - 文本分类（体育/财经等）")
                print("2 - 情感分析（正面/负面）")
                print("3 - 机器翻译（中→英）")
                print("4 - 豆包图文问答")
                choice = input("输入功能编号：").strip()
                
                if choice.lower() == 'exit':
                    print("程序退出！")
                    break
                
                # 豆包API需要额外输入图片URL
                if choice == '4':
                    text = input("请输入要提问的问题：").strip()
                    if not text:
                        print("错误：请输入有效问题！")
                        continue
                    image_url = input("请输入图片URL（可选，直接回车跳过）：").strip()
                    image_url = image_url if image_url else None
                    print(doubao_api_chat(text, image_url))
                else:
                    text = input("请输入待分析文本：").strip()
                    if not text:
                        print("错误：请输入有效文本！")
                        continue
                    
                    if choice == '1':
                        print(text_classification(text))
                    elif choice == '2':
                        print(sentiment_analysis(text))
                    elif choice == '3':
                        print(machine_translation(text))
                    else:
                        print("错误：无效的功能编号！")
            except KeyboardInterrupt:
                print("\n程序被用户中断，退出！")
                break
            except Exception as e:
                print(f"程序运行出错：{str(e)}")