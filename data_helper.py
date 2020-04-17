# Author:duguiming
# Description: 数据处理
# Date: 2020-4-15
import codecs
import math
import random

from utils import create_dico, create_mapping, zero_digits
from bert import tokenization

tokenizer = tokenization.FullTokenizer(vocab_file='bert_model/chinese_L-12_H-768_A-12/vocab.txt',
                                       do_lower_case=True)


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num += 1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
            else:
                word = line.split()
            assert len(word) >= 2, print([word[0]])
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]

    dico = create_dico(tags)
    dico['[SEP]'] = len(dico) + 1
    dico['[CLS]'] = len(dico) + 2

    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def convert_single_example(char_line, tag_to_id, max_seq_length, tokenizer, label_line):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为lb
    """
    text_list = char_line.split(' ')
    label_list = label_line.split(' ')

    tokens = []
    labels = []
    for i, word in enumerate(text_list):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = label_list[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(tag_to_id["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(tag_to_id[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(tag_to_id["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")

    return input_ids, input_mask, segment_ids, label_ids


def prepare_dataset(sentences, max_seq_length, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0].strip() for w in s]
        char_line = ' '.join(string)   # 使用空格把汉字拼起来
        text = tokenization.convert_to_unicode(char_line)

        if train:
            tags = [w[-1] for w in s]
        else:
            tags = ['O' for _ in string]

        labels = ' '.join(tags)     # 使用空格把标签拼起来
        labels = tokenization.convert_to_unicode(labels)

        ids, mask, segment_ids, label_ids = convert_single_example(char_line=text,
                                                                   tag_to_id=tag_to_id,
                                                                   max_seq_length=max_seq_length,
                                                                   tokenizer=tokenizer,
                                                                   label_line=labels)
        data.append([string, segment_ids, ids, mask, label_ids])

    return data


class BatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.arrange_batch(sorted_data[int(i*batch_size) : int((i+1)*batch_size)]))
        return batch_data

    @staticmethod
    def arrange_batch(batch):
        '''
        把batch整理为一个[5, ]的数组
        :param batch:
        :return:
        '''
        strings = []
        segment_ids = []
        chars = []
        mask = []
        targets = []
        for string, seg_ids, char, msk, target in batch:
            strings.append(string)
            segment_ids.append(seg_ids)
            chars.append(char)
            mask.append(msk)
            targets.append(target)
        return [strings, segment_ids, chars, mask, targets]

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, segment_ids, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def input_from_line(line, max_seq_length, tag_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    string = [w[0].strip() for w in line]
    # chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
    #         for w in string]
    char_line = ' '.join(string)  # 使用空格把汉字拼起来
    text = tokenization.convert_to_unicode(char_line)

    tags = ['O' for _ in string]

    labels = ' '.join(tags)  # 使用空格把标签拼起来
    labels = tokenization.convert_to_unicode(labels)

    ids, mask, segment_ids, label_ids = convert_single_example(char_line=text,
                                                               tag_to_id=tag_to_id,
                                                               max_seq_length=max_seq_length,
                                                               tokenizer=tokenizer,
                                                               label_line=labels)
    import numpy as np
    segment_ids = np.reshape(segment_ids,(1, max_seq_length))
    ids = np.reshape(ids, (1, max_seq_length))
    mask = np.reshape(mask, (1, max_seq_length))
    label_ids = np.reshape(label_ids, (1, max_seq_length))
    return [string, segment_ids, ids, mask, label_ids]