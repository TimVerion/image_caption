"""
1.Data generator
    a. Loads vocab.载入词表
    b. Load image features.载入图像预处理特征
    c. provide data for training.不停的提供数据
2. Builds image caption model.
4. Train the model.
"""

import os
import sys
import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
import pprint
import pickle
import numpy as np
import math

input_description_file = "./data/results_20130124.token"
input_img_feature_dir = "./data/feature_extraction_inception_v3"
input_vocab_file = "./data/vocab.txt"
output_dir = "./data/local_run"

if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)


# 使用HParams来设置默认参数
def get_default_params():
    return tf.contrib.training.HParams(
        num_vocab_word_threshold=3,  # 词频过滤阈值
        num_embedding_nodes=32,  # 处理后数据维度
        num_timesteps=10,  # 时间长度
        num_lstm_nodes=[64, 64],  # 每一层的神经元个数
        num_lstm_layers=2,  # lstm的层数
        num_fc_nodes=32,  # 全连接的参数
        batch_size=50,  # 批量训练的个数
        cell_type='lstm',  # 循环神经网络的类型
        clip_lstm_grads=1.0,  # 梯度剪切,反向梯度大小大于1的时候设为1
        learning_rate=0.001,  # 学习率
        keep_prob=0.8,  # dropout的保留率
        log_frequent=100,  # 每隔多少步打印一下信息
        save_frequent=1000,  # 每隔多少次保存一下模型
    )


hps = get_default_params()


# 定义词表--> 过滤单词
class Vocab(object):
    def __init__(self, filename, word_num_threshold):
        self._id_to_word = {}
        self._word_to_id = {}
        # 设置两个特殊的字符
        self._unk = -1
        self._eos = -1
        self._word_num_threshold = word_num_threshold
        self._read_dict(filename)  # 类函数

    def _read_dict(self, filename):
        with gfile.GFile(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            word, occurence = line.strip('\r\n').split('\t')
            occurence = int(occurence)
            # 过滤<UNK>单词和频率较小的单词
            if word != '<UNK>' and occurence < self._word_num_threshold:
                continue
            idx = len(self._id_to_word)
            if word == '<UNK>':
                self._unk = idx
            elif word == '.':
                self._eos = idx
            if idx in self._id_to_word or word in self._word_to_id:
                raise Exception('duplicate words in vocab file')
            self._word_to_id[word] = idx
            self._id_to_word[idx] = word

    @property
    def unk(self):
        return self._unk

    @property
    def eos(self):
        return self._eos

    def word_to_id(self, word):
        return self._word_to_id.get(word, self.unk)

    def id_to_word(self, cur_id):
        return self._id_to_word.get(cur_id, '<UNK>')

    def size(self):
        return len(self._word_to_id)

    def encode(self, sentence):
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split(' ')]
        return word_ids

    def decode(self, sentence_id):
        words = [self.id_to_word(word_id) for word_id in sentence_id]
        return ' '.join(words)


def parse_token_file(token_file):
    """Parses token file."""
    img_name_to_tokens = {}
    with gfile.GFile(token_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        img_id, description = line.strip('\r\n').split('\t')
        img_name, _ = img_id.split('#')
        img_name_to_tokens.setdefault(img_name, [])
        img_name_to_tokens[img_name].append(description)
    return img_name_to_tokens


def convert_token_to_id(img_name_to_tokens, vocab):
    """Converts tokens of each description of imgs to id. """
    img_name_to_token_ids = {}
    for img_name in img_name_to_tokens:
        img_name_to_token_ids.setdefault(img_name, [])
        descriptions = img_name_to_tokens[img_name]
        for description in descriptions:
            token_ids = vocab.encode(description)
            img_name_to_token_ids[img_name].append(token_ids)
    return img_name_to_token_ids


vocab = Vocab(input_vocab_file, hps.num_vocab_word_threshold)
# vocab_size = vocab.size()
# logging.info("vocab_size: %d" % vocab_size)
img_name_to_tokens = parse_token_file(input_description_file)
img_name_to_token_ids = convert_token_to_id(img_name_to_tokens, vocab)# 生成词表


# print(img_name_to_token_ids["6322364581.jpg"])
# exit()
# logging.info("num of all images: %d" % len(img_name_to_tokens))
# pprint.pprint(img_name_to_tokens.keys()[0:10])
# pprint.pprint(img_name_to_tokens['2778832101.jpg'])
# logging.info("num of all images: %d" % len(img_name_to_token_ids))
# pprint.pprint(img_name_to_token_ids.keys()[0:10])
# pprint.pprint(img_name_to_token_ids['2778832101.jpg'])


class ImageCaptionData(object):
    def __init__(self,
                 img_name_to_token_ids,
                 img_feature_dir,
                 num_timesteps,
                 vocab,
                 deterministic=False):
        self._vocab = vocab
        self._all_img_feature_filepaths = []
        for filename in gfile.ListDirectory(img_feature_dir):
            self._all_img_feature_filepaths.append(os.path.join(img_feature_dir, filename))
        # pprint.pprint(self._all_img_feature_filepaths)

        self._img_name_to_token_ids = img_name_to_token_ids
        self._num_timesteps = num_timesteps
        self._indicator = 0
        self._deterministic = deterministic
        self._img_feature_filenames = []
        self._img_feature_data = []
        self._load_img_feature_pickle()
        if not self._deterministic:
            self._random_shuffle()

    def _load_img_feature_pickle(self):
        for filepath in self._all_img_feature_filepaths:
            logging.info("loading %s" % filepath)
            with gfile.GFile(filepath, 'rb') as f:
                filenames, features = pickle.load(f)
                self._img_feature_filenames += filenames
                self._img_feature_data.append(features)
        self._img_feature_data = np.vstack(self._img_feature_data)
        origin_shape = self._img_feature_data.shape
        self._img_feature_data = np.reshape(
            self._img_feature_data, (origin_shape[0], origin_shape[3]))
        self._img_feature_filenames = np.asarray(self._img_feature_filenames)
        print(self._img_feature_data.shape)
        print(self._img_feature_filenames.shape)
        if not self._deterministic:
            self._random_shuffle()

    def size(self):
        return len(self._img_feature_filenames)

    def img_feature_size(self):
        return self._img_feature_data.shape[1]

    def _random_shuffle(self):
        p = np.random.permutation(self.size())
        self._img_feature_filenames = self._img_feature_filenames[p]
        self._img_feature_data = self._img_feature_data[p]

    def _img_desc(self, filenames):
        batch_sentence_ids = []
        batch_weights = [] # 将值为0的标志为0 不用在损失中计算 [1,3,5,0,0] # ->[1,1,1,0.0] 去除噪音
        for filename in filenames:
            token_ids_set = self._img_name_to_token_ids[filename]
            # chosen_token_ids = random.choice(token_ids_set)
            chosen_token_ids = token_ids_set[0]
            chosen_token_length = len(chosen_token_ids)

            weight = [1 for i in range(chosen_token_length)]
            if chosen_token_length >= self._num_timesteps:
                chosen_token_ids = chosen_token_ids[0:self._num_timesteps]
                weight = weight[0:self._num_timesteps]
            else:
                remaining_length = self._num_timesteps - chosen_token_length
                chosen_token_ids += [self._vocab.eos for i in range(remaining_length)]
                weight += [0 for i in range(remaining_length)]
            batch_sentence_ids.append(chosen_token_ids)
            batch_weights.append(weight)
        batch_sentence_ids = np.asarray(batch_sentence_ids)
        batch_weights = np.asarray(batch_weights)
        return batch_sentence_ids, batch_weights

    def next(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self.size():
            if not self._deterministic:
                self._random_shuffle()
            self._indicator = 0
            end_indicator = self._indicator + batch_size
        assert end_indicator <= self.size()

        batch_img_features = self._img_feature_data[self._indicator: end_indicator]
        batch_img_names = self._img_feature_filenames[self._indicator: end_indicator]
        batch_sentence_ids, batch_weights = self._img_desc(batch_img_names)

        self._indicator = end_indicator
        return batch_img_features, batch_sentence_ids, batch_weights, batch_img_names


caption_data = ImageCaptionData(img_name_to_token_ids, input_img_feature_dir, hps.num_timesteps, vocab)
img_feature_dim = caption_data.img_feature_size()
caption_data_size = caption_data.size()
logging.info("img_feature_dim: %d" % img_feature_dim)
logging.info("caption_data_size: %d" % caption_data_size)


# batch_img_features, batch_sentence_ids, batch_weights, batch_img_names = caption_data.next(5)
# pprint.pprint(batch_img_features)
# pprint.pprint(batch_sentence_ids)
# pprint.pprint(batch_weights)
# pprint.pprint(batch_img_names)


def create_rnn_cell(hidden_dim, cell_type):
    if cell_type == 'lstm':
        return tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    elif cell_type == 'gru':
        return tf.contrib.rnn.GRUCell(hidden_dim)
    else:
        raise Exception("%s has not been supported" % cell_type)


def dropout(cell, keep_prob):
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


def get_train_model(hps, vocab_size, img_feature_dim):
    num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size

    img_feature = tf.placeholder(tf.float32, (batch_size, img_feature_dim))
    sentence = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    mask = tf.placeholder(tf.float32, (batch_size, num_timesteps))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step', trainable=False)

    # Sets up the embedding layer.
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope('embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable(
            'embeddings',
            [vocab_size, hps.num_embedding_nodes],
            tf.float32)
        embed_token_ids = tf.nn.embedding_lookup(embeddings, sentence[:, 0:num_timesteps - 1])

    img_feature_embed_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('image_feature_embed', initializer=img_feature_embed_init):
        embed_img = tf.layers.dense(img_feature, hps.num_embedding_nodes)
        embed_img = tf.expand_dims(embed_img, 1)
        embed_inputs = tf.concat([embed_img, embed_token_ids], axis=1)

    # Sets up LSTM network.
    scale = 1.0 / math.sqrt(hps.num_embedding_nodes + hps.num_lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope('lstm_nn', initializer=lstm_init):
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = create_rnn_cell(hps.num_lstm_nodes[i], hps.cell_type)
            cell = dropout(cell, keep_prob)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        initial_state = cell.zero_state(hps.batch_size, tf.float32)
        # rnn_outputs: [batch_size, num_timesteps, hps.num_lstm_node[-1]]
        rnn_outputs, _ = tf.nn.dynamic_rnn(cell,
                                           embed_inputs,
                                           initial_state=initial_state)

    # Sets up the fully-connected layer.
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hps.num_lstm_nodes[-1]])
        fc1 = tf.layers.dense(rnn_outputs_2d, hps.num_fc_nodes, name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        fc1_dropout = tf.nn.relu(fc1_dropout)
        logits = tf.layers.dense(fc1_dropout, vocab_size, name='logits')

    with tf.variable_scope('loss'):
        sentence_flatten = tf.reshape(sentence, [-1])
        mask_flatten = tf.reshape(mask, [-1])
        mask_sum = tf.reduce_sum(mask_flatten)
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=sentence_flatten)
        weighted_softmax_loss = tf.multiply(softmax_loss,
                                            tf.cast(mask_flatten, tf.float32))
        prediction = tf.argmax(logits, 1, output_type=tf.int32)
        correct_prediction = tf.equal(prediction, sentence_flatten)
        correct_prediction_with_mask = tf.multiply(
            tf.cast(correct_prediction, tf.float32),
            mask_flatten)
        accuracy = tf.reduce_sum(correct_prediction_with_mask) / mask_sum
        loss = tf.reduce_sum(weighted_softmax_loss) / mask_sum
        tf.summary.scalar('loss', loss)

    with tf.variable_scope('train_op'):
        tvars = tf.trainable_variables()
        for var in tvars:
            logging.info("variable name: %s" % (var.name))
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), hps.clip_lstm_grads)
        for grad, var in zip(grads, tvars):
            tf.summary.histogram('%s_grad' % (var.name), grad)
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return ((img_feature, sentence, mask, keep_prob),
            (loss, accuracy, train_op),
            global_step)


placeholders, metrics, global_step = get_train_model(hps, vocab_size, img_feature_dim)
img_feature, sentence, mask, keep_prob = placeholders
loss, accuracy, train_op = metrics

summary_op = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=10)

training_steps = 10000

with tf.Session() as sess:
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state(output_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(output_dir, ckpt_name))
        print("restore ok ...")
    writer = tf.summary.FileWriter(output_dir, sess.graph)
    for i in range(training_steps):
        batch_img_features, batch_sentence_ids, batch_weights, _ = caption_data.next(hps.batch_size)
        input_vals = (batch_img_features, batch_sentence_ids, batch_weights, hps.keep_prob)

        feed_dict = dict(zip(placeholders, input_vals))
        fetches = [global_step, loss, accuracy, train_op]

        should_log = (i + 1) % hps.log_frequent == 0
        should_save = (i + 1) % hps.save_frequent == 0
        if should_log:
            fetches += [summary_op]
        outputs = sess.run(fetches, feed_dict)
        global_step_val, loss_val, accuracy_val = outputs[0:3]
        if should_log:
            summary_str = outputs[4]
            writer.add_summary(summary_str, global_step_val)
            # logging.info('Step: %5d, loss: %3.3f, accuracy: %3.3f'% (global_step_val, loss_val, accuracy_val))
            print('Step: %5d, loss: %3.3f, accuracy: %3.3f' % (global_step_val, loss_val, accuracy_val))
        if should_save:
            # logging.info("Step: %d, image caption model saved" % (global_step_val))
            print("Step: %d, image caption model saved" % (global_step_val))
            saver.save(sess, os.path.join(output_dir, "image_caption"), global_step=global_step_val)
