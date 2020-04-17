# Author: duguiming
# Description: 运行程序
# Date: 2020-4-15
import os
import pickle
import argparse

from models.BERT_BiLSTM_CRF import Config, BertBiLSTMCrf
from data_helper import tag_mapping, load_sentences, prepare_dataset, BatchManager
from train_val_test import train, test, demo
from utils import make_path


parser = argparse.ArgumentParser(description='Chinese NER')
parser.add_argument('--mode', type=str, required=True, help='train test or demo')
args = parser.parse_args()


if __name__ == "__main__":
    mode = args.mode
    config = Config()

    # load data
    train_sentences = load_sentences(config.train_path, config.lower, config.zeros)
    dev_sentences = load_sentences(config.dev_path, config.lower, config.zeros)
    test_sentences = load_sentences(config.test_path, config.lower, config.zeros)

    # tags dict
    if not os.path.isfile(config.map_file):
        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(config.map_file, "wb") as f:
            pickle.dump([tag_to_id, id_to_tag], f)
    else:
        with open(config.map_file, "rb") as f:
            tag_to_id, id_to_tag = pickle.load(f)

    config.num_tags = len(tag_to_id)

    train_data = prepare_dataset(
        train_sentences, config.max_seq_len, tag_to_id, config.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, config.max_seq_len, tag_to_id, config.lower
    )
    test_data = prepare_dataset(
        test_sentences, config.max_seq_len, tag_to_id, config.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    train_manager = BatchManager(train_data, config.batch_size)
    dev_manager = BatchManager(dev_data, config.batch_size)
    test_manager = BatchManager(test_data, config.batch_size)

    model = BertBiLSTMCrf(config)
    make_path(config)

    if mode == "train":
        train(model, config, train_manager, dev_manager, id_to_tag)
    elif mode == "test":
        test(model, config, test_manager, id_to_tag)
    else:
        demo(model, config, id_to_tag, tag_to_id)
