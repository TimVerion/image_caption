"""
1. Loads vocab
2. Builds data generator
3. Builds image caption model
4. Train
4. Eval
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


def get_default_params():
    return tf.contrib.training.HParams(
        num_vocab_word_threshold=3,
        num_embedding_nodes=32,
        num_timesteps=10,
        num_lstm_nodes=[64, 64],
        num_lstm_layers=2,
        num_fc_nodes=32,
        batch_size=1,
        cell_type='lstm',
        clip_lstm_grads=1.0,
        learning_rate=0.001,
        keep_prob=0.8,
        log_frequent=100,
        save_frequent=1000,
    )


hps = get_default_params()


class Vocab(object):
    def __init__(self, filename, word_num_threshold):
        self._id_to_word = {}
        self._word_to_id = {}
        self._unk = -1
        self._eos = -1
        self._word_num_threshold = word_num_threshold
        self._read_dict(filename)

    def _read_dict(self, filename):
        with gfile.GFile(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            word, occurence = line.strip('\r\n').split('\t')
            occurence = int(occurence)
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
vocab_size = vocab.size()
logging.info("vocab_size: %d" % vocab_size)

img_name_to_tokens = parse_token_file(input_description_file)
img_name_to_token_ids = convert_token_to_id(img_name_to_tokens, vocab)
# 打印一下所有图片的获取情况
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
        print(origin_shape)
        exit()
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
        batch_weights = []
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
# 验证一下caption_data.next() 方法
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


def eval_get_embedding_for_img(hps, img_feature_dim):
    img_feature = tf.placeholder(tf.float32, (1, img_feature_dim))
    img_feature_embed_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('image_feature_embed', initializer=img_feature_embed_init):
        embed_img = tf.layers.dense(img_feature, hps.num_embedding_nodes)
        embed_img = tf.expand_dims(embed_img, 1)
        return img_feature, embed_img


def eval_embedding_lookup(hps, vocab_size):
    word = tf.placeholder(tf.int32, (1, 1))
    # Sets up the embedding layer.
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope('embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable(
            'embeddings',
            [vocab_size, hps.num_embedding_nodes],
            tf.float32)
        embed_word = tf.nn.embedding_lookup(embeddings, word)
    return word, embed_word


def eval_lstm_single_step(hps, vocab_size):
    embed_input = tf.placeholder(tf.float32, (1, 1, hps.num_embedding_nodes))
    num_lstm_layers = []
    for i in range(hps.num_lstm_layers):
        num_lstm_layers.append(hps.num_lstm_nodes[i])
        num_lstm_layers.append(hps.num_lstm_nodes[i])

    num_hidden_states = sum(num_lstm_layers)
    input_state = tf.placeholder(tf.float32, (1, num_hidden_states))
    unpack_init_state = tf.split(input_state, num_lstm_layers, axis=1)
    input_tuple_state = []
    i = 0
    while i < len(unpack_init_state):
        input_tuple_state.append(
            tf.nn.rnn_cell.LSTMStateTuple(
                unpack_init_state[i], unpack_init_state[i + 1]))
        i += 2
    input_tuple_state = tuple(input_tuple_state)

    scale = 1.0 / math.sqrt(hps.num_embedding_nodes + hps.num_lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope('lstm_nn', initializer=lstm_init):
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = create_rnn_cell(hps.num_lstm_nodes[i], hps.cell_type)
            cell = dropout(cell, 1.0)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        rnn_output, output_tuple_state = tf.nn.dynamic_rnn(
            cell,
            embed_input,
            initial_state=input_tuple_state)
        output_state = []
        for state in output_tuple_state:
            output_state.append(state[0])
            output_state.append(state[1])
        output_state = tf.concat(output_state, axis=1, name="output_state")

    # Sets up the fully-connected layer.
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):
        rnn_output_2d = tf.reshape(rnn_output, [-1, hps.num_lstm_nodes[-1]])
        fc1 = tf.layers.dense(rnn_output_2d, hps.num_fc_nodes, name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, 1.0)
        fc1_dropout = tf.nn.relu(fc1_dropout)
        logits = tf.layers.dense(fc1_dropout, vocab_size, name='logits')

    return embed_input, rnn_output, logits, input_state, output_state, num_hidden_states


img_feature, embed_img = eval_get_embedding_for_img(hps, img_feature_dim)
word, embed_word = eval_embedding_lookup(hps, vocab_size)
embed_input, rnn_output, logits, input_state, output_state, num_hidden_states = eval_lstm_single_step(hps, vocab_size)

summary_op = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=10)

test_examples = 1

with tf.Session() as sess:
    sess.run(init_op)
    logging.info("[*] Reading checkpoint ...")
    ckpt = tf.train.get_checkpoint_state(output_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(output_dir, ckpt_name))
        logging.info("[*] Success Read Checkpoint From %s" % (ckpt_name))
    else:
        raise Exception("[*] Failed load checkpoint")

    for i in range(test_examples):
        single_img_features, single_sentence_ids, single_weights, single_img_name = caption_data.next(hps.batch_size)
        print(single_img_name)

        # pprint.pprint(img_name_to_tokens[single_img_name[0]])
        # pprint.pprint(img_name_to_token_ids[single_img_name[0]])

        embed_img_val = sess.run(embed_img, feed_dict={img_feature: single_img_features})

        state_val = np.zeros((1, num_hidden_states))
        embed_input_val = embed_img_val
        generated_sequence = []

        for j in range(hps.num_timesteps):
            logits_val, state_val = sess.run([logits, output_state],
                                             feed_dict={
                                                 embed_input: embed_input_val,
                                                 input_state: state_val
                                             })
            predicted_word_id = np.argmax(logits_val[0])
            generated_sequence.append(predicted_word_id)
            embed_input_val = sess.run(embed_word,
                                       feed_dict={word: [[predicted_word_id]]})
        pprint.pprint("generated words: ")
        pprint.pprint(generated_sequence)
        pprint.pprint(vocab.decode(generated_sequence))
