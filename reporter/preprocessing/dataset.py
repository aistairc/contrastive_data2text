import itertools
from typing import Tuple

import torch
from torchtext.data import Field, Iterator, RawField, TabularDataset

from reporter.util.config import Config
from reporter.util.constant import (
    N_LONG_TERM,
    N_SHORT_TERM,
    SeqType,
    SpecialToken
)
from reporter.util.conversion import stringify_ric_seqtype


def create_dataset(config: Config, device: torch.device) -> Tuple[Field, Iterator, Iterator, Iterator]:

    fields = dict()
    raw_field = RawField()
    # torchtext 0.3.1
    # AttributeError: 'RawField' object has no attribute 'is_target'
    raw_field.is_target = False
    fields[SeqType.ArticleID.value] = (SeqType.ArticleID.value, raw_field)

    time_field = Field(use_vocab=False, batch_first=True, sequential=False)
    fields['jst_hour'] = (SeqType.Time.value, time_field)

    token_field = \
        Field(use_vocab=True,
              init_token=SpecialToken.BOS.value,
              eos_token=SpecialToken.EOS.value,
              pad_token=SpecialToken.Padding.value,
              unk_token=SpecialToken.Unknown.value) \
        if config.use_init_token_tag \
        else Field(use_vocab=True,
                   eos_token=SpecialToken.EOS.value,
                   pad_token=SpecialToken.Padding.value,
                   unk_token=SpecialToken.Unknown.value)

    fields['processed_tokens'] = (SeqType.Token.value, token_field)

    seqtypes = [SeqType.RawShort, SeqType.RawLong,
                SeqType.MovRefShort, SeqType.MovRefLong,
                SeqType.NormMovRefShort, SeqType.NormMovRefLong,
                SeqType.StdShort, SeqType.StdLong]

    for (ric, seqtype) in itertools.product(config.rics, seqtypes):
        n = N_LONG_TERM \
            if seqtype.value.endswith('long') \
            else N_SHORT_TERM
        price_field = Field(use_vocab=False,
                            fix_length=n,
                            batch_first=True,
                            pad_token=0.0,
                            preprocessing=lambda xs: [float(x) for x in xs],
                            dtype=torch.float)
        key = stringify_ric_seqtype(ric, seqtype)
        fields[key] = (key, price_field)

    train, val, test = \
        TabularDataset.splits(path=str(config.dir_output),
                              format='json',
                              train='alignment-train.json',
                              validation='alignment-valid.json',
                              test='alignment-test.json',
                              fields=fields)

    token_field.build_vocab(train, min_freq=config.token_min_freq)

    batch_size = config.batch_size
    train_iter, val_iter, test_iter = \
        Iterator.splits((train, val, test),
                        batch_sizes=(batch_size, batch_size, batch_size),
                        device=-1 if device.type == 'cpu' else device,
                        repeat=False,
                        sort=False)

    return (token_field, train_iter, val_iter, test_iter)
