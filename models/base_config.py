# AUthor: duguiming
# Description: 基础的配置项目
# Date: 2020-04-15
import os

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class BaseConfig(object):
    # bert模型参数初始化的地方
    init_checkpoint = "bert_model/chinese_L-12_H-768_A-12/bert_model.ckpt"
    bert_config_path = "bert_model/chinese_L-12_H-768_A-12/bert_config.json"

    # 数据的路径
    train_path = os.path.join(base_dir, 'data', 'train.txt')
    dev_path = os.path.join(base_dir, 'data', 'dev.txt')
    test_path = os.path.join(base_dir, 'data', 'test.txt')

    # 存放结果的路径
    map_file = os.path.join(base_dir, 'maps.pkl')
    result_path = os.path.join(base_dir, 'result')
    ckpt_path = os.path.join(base_dir, 'ckpt')
    log_path = os.path.join(base_dir, 'log')
    log_file = os.path.join(log_path, 'train.log')
    checkpoint_path = os.path.join(ckpt_path, 'ner.ckpt')



