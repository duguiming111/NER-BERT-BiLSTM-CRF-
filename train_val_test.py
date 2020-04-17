# Author: duguiming
# Description: 训练、验证和测试
# Date:2020-4-25
import tensorflow as tf
import numpy as np
from tensorflow.contrib.crf import viterbi_decode

from utils import get_logger, test_ner, bio_to_json
from data_helper import input_from_line


def get_feed_dict(model, is_train, batch, config):
    """
    :param is_train: Flag, True for train batch
    :param batch: list train/evaluate data
    :return: structured data to feed
    """
    _, segment_ids, chars, mask, tags = batch
    feed_dict = {
        model.input_ids: np.asarray(chars),
        model.input_mask: np.asarray(mask),
        model.segment_ids: np.asarray(segment_ids),
        model.dropout: 1.0,
    }
    if is_train:
        feed_dict[model.targets] = np.asarray(tags)
        feed_dict[model.dropout] = config.dropout_keep_prob
    return feed_dict


def decode(logits, lengths, matrix, config):
    """
    :param logits: [batch_size, num_steps, num_tags]float32, logits
    :param lengths: [batch_size]int32, real length of each sequence
    :param matrix: transaction matrix for inference
    :return:
    """
    # inference final labels usa viterbi Algorithm
    paths = []
    small = -1000.0
    start = np.asarray([[small]*config.num_tags +[0]])
    for score, length in zip(logits, lengths):
        score = score[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)
        path, _ = viterbi_decode(logits, matrix)

        paths.append(path[1:])
    return paths


def evaluate_(sess, model, data_manager, id_to_tag, config):
    """
    :param sess: session  to run the model
    :param data: list of data
    :param id_to_tag: index to tag name
    :return: evaluate result
    """
    results = []
    trans = model.trans.eval()
    for batch in data_manager.iter_batch():
        strings = batch[0]
        labels = batch[-1]
        feed_dict = get_feed_dict(model, False, batch, config)
        lengths, scores = sess.run([model.lengths, model.logits], feed_dict)
        batch_paths = decode(scores, lengths, trans, config)
        for i in range(len(strings)):
            result = []
            string = strings[i][:lengths[i]]
            gold = [id_to_tag[int(x)] for x in labels[i][1:lengths[i]]]
            pred = [id_to_tag[int(x)] for x in batch_paths[i][1:lengths[i]]]
            for char, gold, pred in zip(string, gold, pred):
                result.append(" ".join([char, gold, pred]))
            results.append(result)
    return results


def evaluate(sess, model, name, data_manager, id_to_tag, logger, config):
    logger.info("evaluate:{}".format(name))
    ner_results = evaluate_(sess, model, data_manager, id_to_tag, config)
    eval_lines = test_ner(ner_results, config.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train(model, config, train_manager, dev_manager, id_to_tag):
    logger = get_logger(config.log_file)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        ckpt = tf.train.get_checkpoint_state(config.ckpt_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logger.info("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        logger.info("start training")
        loss = []
        for i in range(config.epoch):
            for batch in train_manager.iter_batch(shuffle=True):
                feed_dict = get_feed_dict(model, True, batch, config)
                global_step, batch_loss, _ = sess.run([model.global_step, model.loss, model.train_op], feed_dict)

                loss.append(batch_loss)
                if global_step % config.print_per_batch == 0:
                    iteration = global_step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, global_step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []
            best = evaluate(sess, model, "dev",  dev_manager, id_to_tag, logger, config)
            if best:
                saver.save(sess, config.checkpoint_path, global_step=global_step)


def test(model, config, test_manager, id_to_tag):
    logger = get_logger(config.log_file)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        ckpt = tf.train.get_checkpoint_state(config.ckpt_path)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            # saver = tf.train.import_meta_graph('ckpt/ner.ckpt.meta')
            # saver.restore(session, tf.train.latest_checkpoint("ckpt/"))
            saver.restore(sess, ckpt.model_checkpoint_path)
        evaluate(sess, model, 'test', test_manager, id_to_tag, logger, config)


def demo(model, config, id_to_tag, tag_to_id):
    logger = get_logger(config.log_file)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        ckpt = tf.train.get_checkpoint_state(config.ckpt_path)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            # saver = tf.train.import_meta_graph('ckpt/ner.ckpt.meta')
            # saver.restore(session, tf.train.latest_checkpoint("ckpt/"))
            saver.restore(sess, ckpt.model_checkpoint_path)
        while True:
            line = input("input sentence, please:")
            inputs = input_from_line(line, config.max_seq_len, tag_to_id)
            trans = model.trans.eval(sess)
            feed_dict = get_feed_dict(model, False, inputs, config)
            lengths, scores = sess.run([model.lengths, model.logits], feed_dict)
            batch_paths = decode(scores, lengths, trans, config)
            tags = [id_to_tag[idx] for idx in batch_paths[0]]
            result = bio_to_json(inputs[0], tags[1:-1])
            print(result['entities'])


