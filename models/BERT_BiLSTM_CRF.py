# Author: duguiming
# Description: 模型
# Date: 2020-4-15
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers

from models.base_config import BaseConfig
from bert import modeling
import models.rnncell as rnn


class Config(BaseConfig):
    batch_size = 128
    epoch = 100
    print_per_batch = 100
    clip = 5
    dropout_keep_prob = 0.5
    lr = 0.001
    optimizer = 'adam'
    zeros = False
    lower = True

    num_tags = None
    lstm_dim = 200
    max_seq_len = 128
    max_epoch = 100
    steps_check = 100


class BertBiLSTMCrf(object):
    def __init__(self, config):
        self.config = config
        # add placeholders for the model
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)

        self.initializer = initializers.xavier_initializer()

        self.bert_bilstm_crf()

    def bert_bilstm_crf(self):
        # parameter
        used = tf.sign(tf.abs(self.input_ids))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.input_ids)[0]
        self.num_steps = tf.shape(self.input_ids)[-1]
        # bert embedding
        embedding = self.bert_embedding()
        # dropout
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)
        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.config.lstm_dim, self.lengths)
        # logits for tags
        self.logits = self.project_layer(lstm_outputs)
        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        # bert模型参数初始化的地方
        init_checkpoint = self.config.init_checkpoint
        # 获取模型中所有的训练参数。
        tvars = tf.trainable_variables()
        # 加载BERT模型
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        print("**** Trainable Variables ****")
        # 打印加载模型的参数
        train_vars = []
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            else:
                train_vars.append(var)
            print("  name = %s, shape = %s%s", var.name, var.shape,
                  init_string)

        optimizer = self.config.optimizer
        if optimizer == "adam":
            self.opt = tf.train.AdamOptimizer(self.config.lr)
        else:
            raise KeyError
        grads = tf.gradients(self.loss, train_vars)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

        self.train_op = self.opt.apply_gradients(
            zip(grads, train_vars), global_step=self.global_step)

    def bert_embedding(self):
        # load bert embedding
        bert_config = modeling.BertConfig.from_json_file(self.config.bert_config_path)  # 配置文件地址。
        model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        embedding = model.get_sequence_output()
        return embedding

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.config.lstm_dim * 2, self.config.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.config.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.config.lstm_dim * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.config.lstm_dim, self.config.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.config.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.config.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.config.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])],
                axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.config.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.config.num_tags + 1, self.config.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths + 1)
            return tf.reduce_mean(-log_likelihood)