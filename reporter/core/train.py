from logging import Logger
from typing import Dict, List

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torchtext.data import Field, Iterator

from reporter.core.negative import NegativeSampling
from reporter.core.network import EncoderDecoder
from reporter.core.operation import (
    get_latest_closing_vals,
    replace_tags_with_vals
)
from reporter.postprocessing.text import remove_bos
from reporter.util.constant import Code, Phase, SeqType
from reporter.util.conversion import stringify_ric_seqtype
from reporter.util.tool import takeuntil


class RunResult:
    def __init__(self,
                 loss: float,
                 lm_loss: float,
                 article_ids: List[str],
                 neg_info_tuples: List[tuple],
                 gold_sents: List[List[str]],
                 gold_sents_num: List[List[str]],
                 pred_sents: List[List[str]],
                 pred_sents_num: List[List[str]],
                 prob_wins: List[bool]):

        self.loss = loss
        self.lm_loss = lm_loss
        self.article_ids = article_ids
        self.neg_info_tuples = neg_info_tuples
        self.gold_sents = gold_sents
        self.gold_sents_num = gold_sents_num
        self.pred_sents = pred_sents
        self.pred_sents_num = pred_sents_num
        self.prob_wins = prob_wins


def calc_sentence_lengths(batch_tokens, pad_index):
    max_num_tokens, batch_size = batch_tokens.size()
    paddings = torch.ones(max_num_tokens, batch_size).byte()
    paddings[batch_tokens == pad_index] = 0
    paddings = paddings[1:, :]  # cut <s>
    sentence_lengths = torch.sum(paddings, dim=0).float().to(device=batch_tokens.device)
    return sentence_lengths


def run(X: Iterator,
        token_field: Field,
        model: EncoderDecoder,
        optimizer: Dict[SeqType, torch.optim.Optimizer],
        criterion: torch.nn.modules.Module,
        neg_criterion: torch.nn.modules.Module,
        phase: Phase,
        negative: NegativeSampling,
        logger: Logger,
        gen_text=False) -> RunResult:

    if phase in [Phase.Valid, Phase.Test]:
        model.eval()
    else:
        model.train()

    accum_loss = 0.0
    accum_lm_loss = 0.0
    all_article_ids = []
    all_neg_info_tuples = []
    all_gold_sents = []
    all_pred_sents = []
    all_gold_sents_with_number = []
    all_pred_sents_with_number = []
    all_prob_wins = []

    for batch in X:

        article_ids = batch.article_id
        times = batch.time
        tokens = batch.token
        raw_short_field = stringify_ric_seqtype(Code.N225.value, SeqType.RawShort)
        latest_vals = [x for x in getattr(batch, raw_short_field).data[:, 0]]
        raw_long_field = stringify_ric_seqtype(Code.N225.value, SeqType.RawLong)
        latest_closing_vals = get_latest_closing_vals(batch, raw_long_field, times)
        sentence_lengths = calc_sentence_lengths(tokens, pad_index=token_field.vocab.stoi[token_field.pad_token])

        # Forward
        is_token_neg_loss = negative.neg_loss_type == 'token' or negative.neg_loss_type == 'unlikelihood'
        if not(phase == Phase.Train and negative.neg_enabled and is_token_neg_loss):
            lm_loss, pred = model(batch, batch.batch_size, tokens, times, criterion, phase)
            lm_loss = lm_loss.sum() / sentence_lengths.sum()

        if phase == Phase.Train:
            if negative.neg_enabled:
                neg_tokens, neg_train_info_tuples, sample_items = \
                    negative.get_negative_train_tokens(batch.token, article_ids)

                if negative.neg_loss_type == 'sentence':
                    # lm backward independent
                    optimizer.zero_grad()
                    lm_loss.backward()
                    optimizer.step()

                    neg_batch = negative.create_negative_train_batch_sentence_loss(batch, neg_tokens, sample_items)
                    neg_loss = model.neg_train_sentence_loss(
                        batch=neg_batch,
                        mini_batch_size=neg_batch.batch_size,
                        time_embedding=neg_batch.time,
                        pos_tokens=tokens[:, sample_items],
                        neg_tokens=neg_tokens,
                        pad_index=negative.pad_index,
                        neg_criterion=neg_criterion)
                    neg_loss *= negative.neg_loss_ratio

                    # neg backward independent
                    optimizer.zero_grad()
                    neg_loss.backward()
                    optimizer.step()
                    loss = lm_loss + neg_loss  # only for the logging

                    all_neg_info_tuples.extend(neg_train_info_tuples)
                else:  # neg_loss_type: token loss
                    is_UL = negative.neg_loss_type == 'unlikelihood'
                    lm_loss, neg_loss, pred = model.neg_train_token_loss(
                        batch=batch,
                        mini_batch_size=batch.batch_size,
                        time_embedding=batch.time,
                        pos_tokens=tokens,
                        neg_tokens=neg_tokens,
                        pad_index=negative.pad_index,
                        criterion=criterion,
                        neg_criterion=neg_criterion,
                        is_unlikelihood_loss=is_UL,
                    )
                    if negative.neg_loss_type == 'token':
                        # Fix unintended division by dimention number by the library.
                        # https://github.com/pytorch/pytorch/blob/v0.4.1/aten/src/THNN/generic/MultiMarginCriterion.c#L106
                        neg_loss *= 2
                    neg_loss *= negative.neg_loss_ratio
                    loss = lm_loss.sum() + neg_loss
                    loss /= sentence_lengths.sum()

                    lm_loss = lm_loss.sum() / sentence_lengths.sum()  # only for the logging

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                loss = lm_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            loss = lm_loss

        all_article_ids.extend(article_ids)

        i_eos = token_field.vocab.stoi[token_field.eos_token]
        # Recover words from ids removing BOS and EOS from gold sentences for evaluation
        gold_sents = [remove_bos([token_field.vocab.itos[i] for i in takeuntil(i_eos, sent)])
                      for sent in zip(*tokens.cpu().numpy())]
        all_gold_sents.extend(gold_sents)

        pred_sents = [remove_bos([token_field.vocab.itos[i] for i in takeuntil(i_eos, sent)]) for sent in zip(*pred)]
        all_pred_sents.extend(pred_sents)

        if gen_text:
            # generate texts and calculate BLEUs
            z_iter = zip(article_ids, gold_sents, pred_sents, latest_vals, latest_closing_vals)
            for (article_id, gold_sent, pred_sent, latest_val, latest_closing_val) in z_iter:

                bleu = sentence_bleu([gold_sent],
                                     pred_sent,
                                     smoothing_function=SmoothingFunction().method1)

                gold_sent_num = replace_tags_with_vals(gold_sent, latest_closing_val, latest_val)
                all_gold_sents_with_number.append(gold_sent_num)

                pred_sent_num = replace_tags_with_vals(pred_sent, latest_closing_val, latest_val)
                all_pred_sents_with_number.append(pred_sent_num)

                description = \
                    '\n'.join(['=== {} ==='.format(phase.value.upper()),
                               'Article ID: {}'.format(article_id),
                               'Gold (tag): {}'.format(', '.join(gold_sent)),
                               'Gold (num): {}'.format(', '.join(gold_sent_num)),
                               'Pred (tag): {}'.format(', '.join(pred_sent)),
                               'Pred (num): {}'.format(', '.join(pred_sent_num)),
                               'BLEU: {:.5f}'.format(bleu),
                               'Loss: {:.5f}'.format(loss.item()),
                               'Latest: {:.2f}'.format(latest_val),
                               'Closing: {:.2f}'.format(latest_closing_val)])
                logger.info(description)  # TODO: info → debug in release

        # positive vs negative probabilities
        model.eval()
        pos_test_tokens_list, neg_test_batch_list = negative.get_negative_test_batches(batch)
        for pos_test_tokens, neg_test_batch in zip(pos_test_tokens_list, neg_test_batch_list):
            gold_probs, neg_probs = model.neg_test_probs(batch=neg_test_batch,
                                                         mini_batch_size=neg_test_batch.batch_size,
                                                         time_embedding=neg_test_batch.time,
                                                         neg_tokens=neg_test_batch.token,
                                                         pos_tokens=pos_test_tokens,
                                                         pad_index=negative.pad_index)
            all_prob_wins += negative.evaluate_probs_win(neg_test_batch.article_id, gold_probs, neg_probs)
        if phase == Phase.Train:
            model.train()

        accum_loss += loss.item()
        accum_lm_loss += lm_loss.item()

    accum_loss /= float(len(X))
    accum_lm_loss /= float(len(X))

    return RunResult(accum_loss,
                     accum_lm_loss,
                     all_article_ids,
                     all_neg_info_tuples,
                     all_gold_sents,
                     all_gold_sents_with_number,
                     all_pred_sents,
                     all_pred_sents_with_number,
                     all_prob_wins)
