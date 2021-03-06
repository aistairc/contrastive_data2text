from datetime import datetime
from logging import Logger
from pathlib import Path

import toml

from reporter.util.constant import Code


class Span:
    def __init__(self, start: str, end: str):
        self.start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S%z')
        self.end = datetime.strptime(end, '%Y-%m-%d %H:%M:%S%z')


class Config:

    def __init__(self, filename: str):

        self.filename = filename
        config = toml.load(self.filename)

        location = config.get('location', {})
        self.dir_resources = Path(location.get('dir_resources', 'resources'))
        self.dir_logs = Path(location.get('dir_logs', 'logs'))
        self.dir_output = Path(location.get('dir_output', 'output'))

        dataset = config.get('dataset', {})
        self.dest_dataset = self.dir_resources / Path(dataset.get('dest_dataset', 'dataset.pkl'))

        self.neg_rule_file = self.dir_resources / Path(location.get('neg_rule_file', 'rules-to-generate-negative-samples.json'))
        self.neg_rule_tag_classes = dataset.get('neg_rule_tag_classes', ['stated_movement'])

        train = config.get('train', {})
        self.seed = int(train.get('seed', 0))
        self.n_epochs = int(train.get('n_epochs', 10))
        self.batch_size = int(train.get('batch_size', 100))
        self.learning_rate = float(train.get('learning_rate', 1e-4))
        self.token_min_freq = int(train.get('token_min_freq', 1))
        self.rics = sorted(train.get('rics', [Code.N225.value]),
                           key=lambda x: '' if x == Code.N225.value else x)
        self.base_ric = train.get('base_ric', Code.N225.value)
        self.use_standardization = bool(train.get('use_standardization', False))
        self.use_init_token_tag = bool(train.get('use_init_token_tag', True))
        self.patience = int(train.get('patience', 10))

        self.tau_margin = float(train.get('tau_margin', 0.5))
        self.neg_enabled = train.get('neg_enabled', True)
        self.neg_loss_type = train.get('neg_loss_type', 'sentence')
        self.neg_samples_ratio = float(train.get('neg_samples_ratio', 0.5))
        self.neg_loss_ratio = float(train.get('neg_loss_ratio', 1.0))

        enc = config.get('encoder', {})
        self.enc_hidden_size = int(enc.get('enc_hidden_size', 256))
        self.enc_n_layers = int(enc.get('enc_n_layers', 3))
        self.base_ric_hidden_size = int(enc.get('base_ric_hidden_size', 256))
        self.ric_hidden_size = int(enc.get('ric_hidden_size', 64))
        self.use_dropout = bool(enc.get('use_dropout', True))
        self.word_embed_size = int(enc.get('word_embed_size', 128))
        self.time_embed_size = int(enc.get('time_embed_size', 64))

        dec = config.get('decoder', {})
        self.dec_hidden_size = int(dec.get('dec_hidden_size', 256))

    def write_log(self, logger: Logger):
        s = '\n'.join(['load configuration from: {}'.format(self.filename),
                       'seed: {}'.format(self.seed),
                       'n_epochs: {}'.format(self.n_epochs),
                       'batch_size: {}'.format(self.batch_size),
                       'learning_rate: {}'.format(self.learning_rate),
                       'neg_enabled: {}'.format(self.neg_enabled),
                       'neg_samples_ratio: {}'.format(self.neg_samples_ratio),
                       'neg_loss_ratio: {}'.format(self.neg_loss_ratio),
                       'neg_loss_type: {}'.format(self.neg_loss_type),
                       'neg_rule_tag_classes: {}'.format(self.neg_rule_tag_classes),
                       'tau_margin: {}'.format(self.tau_margin),
                       'token_min_freq: {}'.format(self.token_min_freq),
                       'rics: {}'.format(self.rics),
                       'base_ric: {}'.format(self.base_ric),
                       'use_standardization: {}'.format(self.use_standardization),
                       'use_init_token_tag: {}'.format(self.use_init_token_tag),
                       'patience: {}'.format(self.patience),
                       'enc_hidden_size: {}'.format(self.enc_hidden_size),
                       'enc_n_layers: {}'.format(self.enc_n_layers),
                       'base_ric_hidden_size: {}'.format(self.base_ric_hidden_size),
                       'ric_hidden_size: {}'.format(self.ric_hidden_size),
                       'use_dropout: {}'.format(self.use_dropout),
                       'word_embed_size: {}'.format(self.word_embed_size),
                       'time_embed_size: {}'.format(self.time_embed_size)])
        logger.info(s)

    @property
    def precise_method_name(self) -> str:
        return ''
